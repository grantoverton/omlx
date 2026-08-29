"""Enable DFlash prefix caching for GLM-5.3 (glm5_next) targets.

The pinned dflash snapshot codec supports bare ``KVCache``,
``RotatingKVCache`` and ``RecurrentRollbackCache``. GLM's DSA layers use
``CacheList(KVCache, PoolingCache)``, so the GLM adapter (PR #3261) declared
``supports_prefix_snapshot=False`` and scrubbed every request to a forced
miss — a full re-prefill (~46 s at 20k tokens) on every DFlash request while
AR/MTP warm-start in ~5 s.

This patch extends the codec instead:

* ``fa`` slot   <- the CacheList's KVCache member ``(keys, values, offset)``.
  Same shape as a bare KVCache entry, so the codec's offset validation and
  the snapshot ``nbytes`` walk hold unchanged.
* ``gdn`` slot  <- the PoolingCache member's 5-tuple ``state``
  ``(buf_kv, buf_gate, pooled, prev_win_kv, prev_win_gate)`` — arrays or
  ``None`` only, so the flat ``nbytes`` walk stays valid.

Hydration dispatches on the *template* entry type (built by the adapter's
``make_cache``), reconstructs the members with the template's own classes and
ratio, and lets ``PoolingCache.state``'s setter rebuild internal buffers and
reset the undo log — the same restore path oMLX's shared prompt cache already
exercises for AR/MTP serving.

Finally the patch flips ``supports_prefix_snapshot`` to ``True`` on the GLM
target ops and retires the forced-miss prefill bridge (its chunk loop becomes
legitimately enabled by the real capability).

Parity gates for this change live in ``glm_cache_parity.py``: repeat-prompt
output must be byte-identical to the cold run at temp 0, with warm TTFT.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Optional

import mlx.core as mx

logger = logging.getLogger(__name__)


def _is_glm_cachelist(entry: Any) -> bool:
    caches = getattr(entry, "caches", None)
    if type(entry).__name__ != "CacheList" or caches is None:
        return False
    if len(caches) != 2:
        return False
    return (
        type(caches[0]).__name__ == "KVCache"
        and type(caches[1]).__name__ == "PoolingCache"
    )


def install_glm5_dflash_prefix_cache() -> bool:
    import dflash_mlx.cache.codecs as codecs
    import dflash_mlx.engine.spec_epoch as spec_epoch
    from mlx_lm.models.cache import KVCache, RotatingKVCache
    from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache

    if getattr(codecs.serialize_target_cache, "_omlx_glm5_prefix", False):
        return False

    _clone = codecs._clone_array

    # ------------------------------------------------------------------ ser
    def serialize_target_cache(
        target_cache: list[Any],
        *,
        clone: bool = True,
    ):
        grab = _clone if clone else (lambda a: a)
        fa: list[Any] = []
        gdn: list[Any] = []
        for layer_idx, entry in enumerate(target_cache):
            if isinstance(entry, RecurrentRollbackCache):
                fa.append(None)
                gdn.append(tuple(grab(a) for a in entry.cache))
            elif isinstance(entry, RotatingKVCache):
                keys = getattr(entry, "keys", None)
                values = getattr(entry, "values", None)
                if keys is None or values is None:
                    fa.append(None)
                    gdn.append(None)
                else:
                    fa.append(
                        (grab(keys), grab(values), int(entry.offset), int(entry._idx))
                    )
                    gdn.append(None)
            elif isinstance(entry, KVCache):
                state = entry.state
                if state is None or state[0] is None:
                    fa.append(None)
                    gdn.append(None)
                else:
                    k, v = state
                    fa.append((grab(k), grab(v), int(entry.offset)))
                    gdn.append(None)
            elif _is_glm_cachelist(entry):
                kv_member, pool_member = entry.caches
                kv_state = kv_member.state
                if kv_state is None or kv_state[0] is None:
                    fa.append(None)
                    gdn.append(None)
                else:
                    k, v = kv_state
                    fa.append((grab(k), grab(v), int(kv_member.offset)))
                    gdn.append(
                        tuple(
                            grab(x) if isinstance(x, mx.array) else None
                            for x in pool_member.state
                        )
                    )
            else:
                raise TypeError(
                    f"Cache entry type {type(entry).__name__} at layer {layer_idx} "
                    "is not supported for prefix-cache serialization."
                )
        return tuple(fa), tuple(gdn)

    # ------------------------------------------------------------------ hyd
    def hydrate_target_cache(snapshot, template_cache: list[Any]) -> list[Any]:
        if len(template_cache) != len(snapshot.fa_states):
            raise ValueError(
                f"Template cache length {len(template_cache)} != "
                f"snapshot layer count {len(snapshot.fa_states)}"
            )
        codecs._validate_snapshot_cache_prefix_len(
            snapshot.fa_states,
            prefix_len=snapshot.prefix_len,
        )
        result: list[Any] = []
        for i, tmpl in enumerate(template_cache):
            fa_state = snapshot.fa_states[i]
            gdn_state = snapshot.gdn_states[i]

            if isinstance(tmpl, RecurrentRollbackCache):
                if gdn_state is None:
                    raise ValueError(f"Snapshot missing GDN state at layer {i}")
                new_cache = RecurrentRollbackCache(
                    size=len(tmpl.cache),
                    conv_kernel_size=tmpl.conv_kernel_size,
                )
                new_cache.cache = list(gdn_state)
                result.append(new_cache)
            elif isinstance(tmpl, KVCache) and not _is_glm_cachelist(tmpl):
                if fa_state is None:
                    raise ValueError(f"Snapshot missing FA state at layer {i}")
                k, v, offset = fa_state[:3]
                if int(k.shape[2]) != int(offset) or int(v.shape[2]) != int(offset):
                    raise ValueError(
                        f"Snapshot FA arrays at layer {i} are not exact-length "
                        f"(keys={int(k.shape[2])}, values={int(v.shape[2])}, "
                        f"offset={int(offset)}); cannot adopt"
                    )
                new_cache = KVCache()
                new_cache.keys = k
                new_cache.values = v
                new_cache.offset = offset
                result.append(new_cache)
            elif isinstance(tmpl, RotatingKVCache):
                if fa_state is None:
                    raise ValueError(
                        f"Snapshot missing rotating FA state at layer {i}"
                    )
                if len(fa_state) != 4:
                    raise ValueError(
                        f"Snapshot missing rotating FA ring index at layer {i}"
                    )
                k, v, offset = fa_state[:3]
                new_cache = RotatingKVCache(
                    max_size=int(tmpl.max_size),
                    keep=int(tmpl.keep),
                )
                new_cache.keys = _clone(k)
                new_cache.values = _clone(v)
                new_cache.offset = int(offset)
                new_cache._idx = int(fa_state[3])
                result.append(new_cache)
            elif _is_glm_cachelist(tmpl):
                if fa_state is None:
                    raise ValueError(
                        f"Snapshot missing DSA KV state at layer {i}"
                    )
                if gdn_state is None:
                    raise ValueError(
                        f"Snapshot missing DSA pooling state at layer {i}"
                    )
                k, v, offset = fa_state[:3]
                if int(k.shape[2]) != int(offset) or int(v.shape[2]) != int(offset):
                    raise ValueError(
                        f"Snapshot DSA KV arrays at layer {i} are not "
                        f"exact-length (keys={int(k.shape[2])}, "
                        f"values={int(v.shape[2])}, offset={int(offset)})"
                    )
                kv_tmpl, pool_tmpl = tmpl.caches
                new_kv = type(kv_tmpl)()
                new_kv.keys = k
                new_kv.values = v
                new_kv.offset = int(offset)
                new_pool = type(pool_tmpl)(int(pool_tmpl.ratio))
                new_pool.state = tuple(gdn_state)
                result.append(type(tmpl)(new_kv, new_pool))
            else:
                raise TypeError(
                    f"Cannot hydrate cache of type {type(tmpl).__name__} at layer {i}"
                )
        return result

    serialize_target_cache._omlx_glm5_prefix = True  # type: ignore[attr-defined]
    hydrate_target_cache._omlx_glm5_prefix = True  # type: ignore[attr-defined]
    serialize_target_cache._omlx_original = codecs.serialize_target_cache  # type: ignore[attr-defined]
    hydrate_target_cache._omlx_original = codecs.hydrate_target_cache  # type: ignore[attr-defined]

    codecs.serialize_target_cache = serialize_target_cache
    codecs.hydrate_target_cache = hydrate_target_cache
    # spec_epoch imported hydrate by name at module load; rebind it there too.
    spec_epoch.hydrate_target_cache = hydrate_target_cache

    # ------------------------------------------------- capability flip
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps

    orig_caps = Glm5NextTargetOps.capabilities_for
    if not getattr(orig_caps, "_omlx_glm5_prefix", False):

        def capabilities_for(self, target_model):
            caps = orig_caps(self, target_model)
            return replace(caps, supports_prefix_snapshot=True)

        capabilities_for._omlx_glm5_prefix = True  # type: ignore[attr-defined]
        capabilities_for._omlx_original = orig_caps  # type: ignore[attr-defined]
        Glm5NextTargetOps.capabilities_for = capabilities_for

    # ------------------------------------------------- retire forced-miss bridge
    current = spec_epoch.SpeculativeSession._run_prefill_events
    if getattr(current, "_omlx_glm5_chunk_bridge", False):
        original = getattr(current, "_omlx_original", None)
        if original is not None:
            spec_epoch.SpeculativeSession._run_prefill_events = original
            logger.info(
                "GLM prefix cache: retired forced-miss prefill bridge "
                "(chunking now enabled by the real capability)"
            )

    logger.info("GLM DFlash prefix-cache serialization installed")
    return True
