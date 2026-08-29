# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 target adapter for the pinned dflash-mlx runtime.

The published ``incoai/GLM-5.3-Flash-DFlash2`` checkpoint already uses the
generic :class:`dflash_mlx.model.DFlash2DraftModel`.  The missing piece is the
target contract: GLM-5.3 is loaded through mlx-vlm, carries multi-stream
hyper-connection (MHC) hidden states, and combines recurrent KDA caches with
``CacheList(KVCache, PoolingCache)`` sparse-attention caches.

This module extends dflash-mlx through its backend registry and target-loader
seam.  It intentionally fails closed:

* only ``glm5_next`` targets paired with a DFlash2 checkpoint are accepted;
* captured MHC states are contracted before reaching the drafter, matching
  SGLang's official GLM-5.3 DFlash capture contract;
* recurrent verification uses bounded replay through GLM's exact vector-gate
  recurrence (the generic innovation tape drifts at tail-ULP after repeated
  GLM rejection cycles);
* composite DSA caches are trimmed explicitly after rejected draft tokens;
* prefix snapshots and tree verification remain disabled until codecs for the
  composite GLM cache are independently parity-proven.

The installer is process-global and idempotent, like the existing Laguna
adapter.  ``restore_glm5_dflash_class_patches`` is called by the shared DFlash
lifecycle restore so a later normal/MTP GLM load cannot inherit speculative
class hooks.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlx.core as mx
from dflash_mlx.engine.target_ops import TargetCapabilities
from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache

_BACKEND_PATH = "omlx.patches.dflash_glm5:Glm5NextTargetOps"
_ORIGINAL_TARGET_LOADER: Any | None = None
_ORIGINAL_LINEAR_CALLS: dict[type, Any] = {}
_ORIGINAL_PREFILL_RUNNER: Any | None = None


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _model_type(model: Any) -> str:
    value = getattr(model, "model_type", None)
    if value is not None:
        return str(value).lower()
    config = getattr(model, "config", None)
    value = _config_value(config, "model_type")
    if value is not None:
        return str(value).lower()
    language_model = getattr(model, "language_model", None)
    args = getattr(language_model, "args", None)
    return str(getattr(args, "model_type", "")).lower()


def _is_glm5_config(config: dict[str, Any]) -> bool:
    return str(config.get("model_type", "")).lower() == "glm5_next"


def _contract_mhc_hidden(hidden: mx.array) -> mx.array:
    """Contract GLM's ``[B, T, hc_mult, H]`` residual streams for DFlash."""
    if hidden.ndim == 4:
        return hidden.mean(axis=2)
    if hidden.ndim != 3:
        raise ValueError(f"Unexpected GLM hidden-state rank: {hidden.ndim}")
    return hidden


def validate_glm5_dflash_pair(
    target_model: Any,
    draft_model: Any,
    draft_meta: Any,
) -> None:
    """Reject an unproven GLM target/draft pairing before generation starts."""
    if _model_type(target_model) != "glm5_next":
        return

    config = draft_meta.get("config", {}) if isinstance(draft_meta, dict) else {}
    architectures = tuple(str(v) for v in (config.get("architectures") or ()))
    if "DFlash2DraftModel" not in architectures or not bool(
        getattr(draft_model, "is_dflash2", False)
    ):
        raise ValueError("GLM-5.3 DFlash requires a DFlash2DraftModel checkpoint")

    language_model = getattr(target_model, "language_model", None)
    target_args = getattr(language_model, "args", None)
    target_inner = getattr(language_model, "model", None)
    draft_args = getattr(draft_model, "args", None)
    if target_args is None or target_inner is None or draft_args is None:
        raise ValueError("GLM-5.3 DFlash target/draft metadata is incomplete")

    checks = (
        ("hidden_size", int(getattr(target_args, "hidden_size", 0))),
        ("vocab_size", int(getattr(target_args, "vocab_size", 0))),
        ("num_target_layers", len(getattr(target_inner, "layers", ()))),
    )
    for draft_name, target_value in checks:
        draft_value = int(getattr(draft_args, draft_name, 0) or 0)
        if draft_value != target_value:
            raise ValueError(
                f"GLM-5.3 DFlash {draft_name} mismatch: "
                f"draft={draft_value}, target={target_value}"
            )

    target_layer_ids = [int(v) for v in getattr(draft_model, "target_layer_ids", ())]
    if (
        not target_layer_ids
        or target_layer_ids != sorted(set(target_layer_ids))
        or target_layer_ids[0] < 0
        or target_layer_ids[-1] >= len(target_inner.layers)
    ):
        raise ValueError(
            "GLM-5.3 DFlash target_layer_ids must be unique, increasing, and "
            "inside the target layer range"
        )

    if (
        not bool(getattr(target_args, "mhc", False))
        or int(getattr(target_args, "hc_mult", 0) or 0) <= 0
    ):
        raise ValueError("GLM-5.3 DFlash requires the checkpoint's MHC target")


def _install_glm5_recurrent_hook(linear_attn: Any) -> None:
    """Retain verify inputs so rejected KDA state can be replayed exactly.

    The target forward itself remains the vendored GLM implementation. This is
    important: a copied verify implementation can silently drift whenever the
    GLM projection or recurrence kernels change. The wrapper records only the
    already-computed layer input and delegates all arithmetic unchanged.
    """
    cls = type(linear_attn)
    if cls in _ORIGINAL_LINEAR_CALLS:
        return
    original_call = cls.__call__
    _ORIGINAL_LINEAR_CALLS[cls] = original_call

    def speculative_call(
        self,
        inputs: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        if not isinstance(cache, RecurrentRollbackCache) or not getattr(
            cache, "_armed", False
        ):
            return original_call(self, inputs, mask=mask, cache=cache)
        output = original_call(self, inputs, mask=mask, cache=cache)
        cache._omlx_glm5_verify = (self, inputs, mask)
        return output

    speculative_call._omlx_dflash_glm5 = True  # type: ignore[attr-defined]
    cls.__call__ = speculative_call


def _install_glm5_prefill_chunking_bridge() -> bool:
    """Chunk GLM prefill without claiming or activating snapshot support.

    The pinned dflash runtime currently uses ``supports_prefix_snapshot`` for
    two independent decisions: whether cache snapshots are legal *and* whether
    cold prefill is split by ``prefill_step_size``.  GLM's composite DSA cache
    is intentionally not serializable, but that must not turn a long prompt
    into one monolithic target forward.

    This scoped bridge temporarily enables only the runtime's chunk loop while
    sanitizing every snapshot-bearing request field.  It then rewrites the
    returned prefill result back to ``supports_prefix_snapshot=False``, so
    decode and generation-snapshot paths continue to see the truthful target
    capability.  Non-GLM sessions take the original method unchanged.
    """
    global _ORIGINAL_PREFILL_RUNNER

    from dflash_mlx.engine.spec_epoch import SpeculativeSession

    current = SpeculativeSession._run_prefill_events
    if getattr(current, "_omlx_glm5_chunk_bridge", False):
        return False
    _ORIGINAL_PREFILL_RUNNER = current

    def run_prefill_chunked(self, *, request, state, yield_pause):
        if (
            getattr(getattr(self, "target_ops", None), "backend_name", "")
            != "glm5_next"
        ):
            return current(
                self,
                request=request,
                state=state,
                yield_pause=yield_pause,
            )

        # Session.open() already received the real False capability, so it
        # cannot hydrate a prefix snapshot.  Scrub the request as a second,
        # local fail-closed boundary before borrowing the chunk loop.
        safe_request = replace(
            request,
            prefix_snapshot=None,
            snapshot_service=None,
            stable_prefix_len=None,
            prefix_cache_active=False,
            publish_generation_snapshot=False,
            prefix_hit_kind="miss",
        )

        def iterate():
            previous = bool(self.supports_prefix_snapshot)
            self.supports_prefix_snapshot = True
            try:
                result = yield from current(
                    self,
                    request=safe_request,
                    state=state,
                    yield_pause=yield_pause,
                )
                return replace(result, supports_prefix_snapshot=False)
            finally:
                self.supports_prefix_snapshot = previous

        return iterate()

    run_prefill_chunked._omlx_glm5_chunk_bridge = True  # type: ignore[attr-defined]
    run_prefill_chunked._omlx_original = current  # type: ignore[attr-defined]
    SpeculativeSession._run_prefill_events = run_prefill_chunked
    return True


def restore_glm5_dflash_class_patches() -> int:
    """Restore every process-global GLM DFlash runtime hook."""
    global _ORIGINAL_PREFILL_RUNNER

    restored = 0
    for cls, original_call in tuple(_ORIGINAL_LINEAR_CALLS.items()):
        cls.__call__ = original_call
        restored += 1
    _ORIGINAL_LINEAR_CALLS.clear()

    if _ORIGINAL_PREFILL_RUNNER is not None:
        try:
            from dflash_mlx.engine.spec_epoch import SpeculativeSession

            current = SpeculativeSession._run_prefill_events
            if getattr(current, "_omlx_glm5_chunk_bridge", False):
                SpeculativeSession._run_prefill_events = _ORIGINAL_PREFILL_RUNNER
                restored += 1
        finally:
            _ORIGINAL_PREFILL_RUNNER = None
    return restored


class Glm5NextTargetOps:
    """dflash-mlx target contract for GLM-5.3's KDA/DSA text backbone."""

    backend_name = "glm5_next"

    def model_type(self, target_model: Any) -> str:
        return _model_type(target_model)

    def supports_model(self, target_model: Any) -> bool:
        if self.model_type(target_model) != "glm5_next":
            return False
        try:
            inner = self.text_model(target_model)
        except AttributeError:
            return False
        return (
            hasattr(inner, "layers")
            and hasattr(inner, "embed_tokens")
            and hasattr(inner, "fa_idx")
            and hasattr(inner, "ssm_idx")
        )

    def family(self, target_model: Any) -> str:
        del target_model
        return "glm5_next_kda_dsa"

    def capabilities_for(self, target_model: Any) -> TargetCapabilities:
        del target_model
        return TargetCapabilities(
            supports_dflash=True,
            supports_recurrent_rollback=True,
            supports_kv_trim=True,
            # dflash-mlx's snapshot codec only supports bare KVCache and its
            # own recurrent cache. GLM DSA uses CacheList(KV, Pooling).
            supports_prefix_snapshot=False,
            supports_rotating_cache_snapshot=False,
            supports_shared_kv=False,
            supports_target_hidden_capture=True,
            # The Qwen verify-qmm kernels have not passed GLM parity gates.
            supports_verify_linear=False,
            supports_full_context_draft_layers=False,
            supports_tree_verify=False,
        )

    def supports_tree_cache(self, cache_entries: list[Any]) -> bool:
        del cache_entries
        return False

    def text_wrapper(self, target_model: Any) -> Any:
        wrapper = getattr(target_model, "language_model", None)
        if wrapper is None or not hasattr(wrapper, "model"):
            raise AttributeError(
                f"Unsupported GLM-5.3 model wrapper: {type(target_model)!r}"
            )
        return wrapper

    def text_model(self, target_model: Any) -> Any:
        return self.text_wrapper(target_model).model

    def embed_tokens(self, target_model: Any) -> Any:
        return self.text_model(target_model).embed_tokens

    def logits_from_hidden(
        self, target_model: Any, hidden_states: mx.array
    ) -> mx.array:
        from mlx_vlm.models.glm5_next.linear import linear_forward

        wrapper = self.text_wrapper(target_model)
        if bool(getattr(wrapper.args, "tie_word_embeddings", False)):
            return wrapper.model.embed_tokens.as_linear(hidden_states)
        return linear_forward(wrapper.lm_head, hidden_states)

    def make_cache(
        self,
        target_model: Any,
        *,
        enable_speculative_linear_cache: bool,
        quantize_kv_cache: bool = False,
        target_fa_window: int | None = None,
    ) -> list[Any]:
        if not enable_speculative_linear_cache:
            raise ValueError("GLM-5.3 DFlash requires recurrent rollback caches")
        if quantize_kv_cache:
            raise ValueError("GLM-5.3 DFlash target KV quantization is unproven")
        if target_fa_window is not None and int(target_fa_window) > 0:
            raise ValueError("GLM-5.3 DFlash does not support target_fa_window")

        wrapper = self.text_wrapper(target_model)
        caches = list(wrapper.make_cache())
        inner = wrapper.model
        if len(caches) != len(inner.layers):
            raise ValueError("GLM-5.3 target cache/layer count mismatch")
        for index, layer in enumerate(inner.layers):
            if getattr(layer, "is_linear", False):
                conv_kernel = int(layer.self_attn.conv_kernel_size)
                caches[index] = RecurrentRollbackCache(
                    size=2, conv_kernel_size=conv_kernel
                )
        return caches

    def install_speculative_hooks(self, target_model: Any) -> None:
        inner = self.text_model(target_model)
        for layer in inner.layers:
            if getattr(layer, "is_linear", False):
                _install_glm5_recurrent_hook(layer.self_attn)

    def forward_with_hidden_capture(
        self,
        target_model: Any,
        *,
        input_ids: mx.array | None = None,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
        capture_layer_ids: set[int] | None = None,
        logits_last_only: bool = False,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        from mlx_vlm.models.base import create_attention_mask, create_ssm_mask

        inner = self.text_model(target_model)
        h = (
            input_embeddings
            if input_embeddings is not None
            else inner.embed_tokens(input_ids)
        )
        if cache is None:
            cache = [None] * len(inner.layers)
        elif len(cache) != len(inner.layers):
            raise ValueError("GLM-5.3 cache/layer count mismatch")

        fa_cache = cache[inner.fa_idx]
        fa_mask = create_attention_mask(
            h,
            fa_cache[0] if fa_cache is not None else None,
            return_array=True,
        )
        ssm_mask = create_ssm_mask(h, cache[inner.ssm_idx])
        h = mx.contiguous(
            mx.broadcast_to(
                h[:, :, None, :],
                (h.shape[0], h.shape[1], inner.hc_mult, h.shape[2]),
            )
        )

        capture_all = capture_layer_ids is None
        if capture_all:
            captured: list[mx.array] | dict[int, mx.array] = [_contract_mhc_hidden(h)]
        else:
            capture_layer_ids = set(capture_layer_ids)
            captured = {0: _contract_mhc_hidden(h)} if 0 in capture_layer_ids else {}

        for layer_index, (layer, layer_cache) in enumerate(
            zip(inner.layers, cache, strict=True)
        ):
            mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
            h = layer(h, mask=mask, cache=layer_cache)
            capture_key = layer_index + 1
            if capture_all:
                captured.append(_contract_mhc_hidden(h))
            elif capture_layer_ids is not None and capture_key in capture_layer_ids:
                captured[capture_key] = _contract_mhc_hidden(h)

        normalized = inner.norm(_contract_mhc_hidden(h))
        if logits_last_only and isinstance(captured, dict):
            captured[-1] = normalized
        logits_hidden = normalized[:, -1:, :] if logits_last_only else normalized
        return self.logits_from_hidden(target_model, logits_hidden), captured

    def verify_block(
        self,
        *,
        target_model: Any,
        verify_ids: mx.array,
        target_cache: list[Any],
        capture_layer_ids: set[int] | None = None,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        if int(verify_ids.shape[1]) <= 0:
            raise ValueError("verify block must contain at least one token")
        # PoolingCache and any future rotating component only retain their
        # cross-boundary undo record while this thread-local gate is armed.
        from ..patches.mlx_lm_mtp import cache_rollback

        cache_rollback.apply()
        cache_rollback.set_undo_armed(True)
        try:
            return self.forward_with_hidden_capture(
                target_model,
                input_ids=verify_ids,
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
        finally:
            cache_rollback.set_undo_armed(False)

    def verify_tree_block(
        self,
        *,
        target_model: Any,
        tree_inputs: Any,
        target_cache: list[Any],
        capture_layer_ids: set[int] | None = None,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        del target_model, tree_inputs, target_cache, capture_layer_ids
        raise NotImplementedError("GLM-5.3 DFlash tree verification is unproven")

    def restore_after_tree_acceptance(
        self, cache_entries: list[Any], *, accepted_tree_indices: list[int]
    ) -> int:
        del cache_entries, accepted_tree_indices
        raise NotImplementedError("GLM-5.3 DFlash tree verification is unproven")

    def extract_context_feature(
        self,
        captured_dict: dict[int, mx.array] | list[mx.array],
        target_layer_ids: list[int],
    ) -> mx.array:
        return mx.concatenate(
            [captured_dict[int(layer_id) + 1] for layer_id in target_layer_ids],
            axis=-1,
        )

    def arm_rollback(self, cache_entries: list[Any], *, prefix_len: int) -> None:
        for cache_entry in cache_entries:
            if isinstance(cache_entry, RecurrentRollbackCache):
                cache_entry.arm_rollback(prefix_len=prefix_len)

    @staticmethod
    def _clear_glm_recurrent_transients(cache_entry: Any) -> None:
        if hasattr(cache_entry, "_omlx_glm5_verify"):
            delattr(cache_entry, "_omlx_glm5_verify")
        cache_entry.clear_transients()

    @staticmethod
    def _clear_composite_undo(cache_entry: Any) -> None:
        """Drop accepted verify-block undo arrays retained by PoolingCache."""
        for component in getattr(cache_entry, "caches", ()):
            if hasattr(component, "_undo"):
                component._undo = None
            if hasattr(component, "_undo_chain"):
                component._undo_chain = False

    @classmethod
    def _rollback_glm_recurrent(
        cls,
        cache_entry: RecurrentRollbackCache,
        *,
        accepted_steps: int,
    ) -> None:
        snapshot = getattr(cache_entry, "_snapshot", None)
        verify = getattr(cache_entry, "_omlx_glm5_verify", None)
        if snapshot is None or verify is None:
            cls._clear_glm_recurrent_transients(cache_entry)
            raise RuntimeError("GLM-5.3 recurrent rollback state is missing")

        attention, inputs, mask = verify
        accepted_steps = max(0, min(int(accepted_steps), int(inputs.shape[1])))
        cache_entry.cache = list(snapshot)
        if accepted_steps > 0:
            step_mask = (
                mask[:, :accepted_steps]
                if isinstance(mask, mx.array) and mask.ndim == 2
                else None
            )
            original_call = _ORIGINAL_LINEAR_CALLS.get(type(attention))
            if original_call is None:
                cls._clear_glm_recurrent_transients(cache_entry)
                raise RuntimeError("GLM-5.3 recurrent target hook is not installed")
            # One bounded native call recreates both conv and KDA state using
            # exactly the target implementation and the accepted-width shape.
            original_call(
                attention,
                inputs[:, :accepted_steps],
                mask=step_mask,
                cache=cache_entry,
            )
        cls._clear_glm_recurrent_transients(cache_entry)

    def restore_after_acceptance(
        self,
        cache_entries: list[Any],
        *,
        target_len: int,
        acceptance_length: int,
        drafted_tokens: int = 0,
    ) -> int:
        started = time.perf_counter_ns()
        changed = False
        fully_accepted = acceptance_length == drafted_tokens
        for cache_entry in cache_entries:
            if isinstance(cache_entry, RecurrentRollbackCache):
                if fully_accepted:
                    self._clear_glm_recurrent_transients(cache_entry)
                else:
                    self._rollback_glm_recurrent(
                        cache_entry,
                        accepted_steps=int(acceptance_length) + 1,
                    )
                changed = True
                continue

            # GLM DSA cache entries are CacheList(KVCache, PoolingCache).
            try:
                kv_cache = cache_entry[0]
            except (IndexError, TypeError):
                kv_cache = None
            offset = int(getattr(kv_cache, "offset", 0) or 0)
            trim_count = max(0, offset - int(target_len))
            if trim_count <= 0:
                if fully_accepted:
                    self._clear_composite_undo(cache_entry)
                continue
            trim = getattr(cache_entry, "trim", None)
            if not callable(trim):
                raise RuntimeError(
                    f"GLM-5.3 DFlash cannot trim {type(cache_entry).__name__}"
                )
            trimmed = int(trim(trim_count))
            if trimmed != trim_count:
                raise RuntimeError(
                    "GLM-5.3 DFlash composite-cache rollback failed: "
                    f"requested={trim_count}, trimmed={trimmed}"
                )
            self._clear_composite_undo(cache_entry)
            changed = True
        return time.perf_counter_ns() - started if changed else 0

    def cleanup_generation_caches(
        self, target_cache: list[Any], draft_cache: list[Any]
    ) -> None:
        for entry in target_cache:
            if isinstance(entry, RecurrentRollbackCache):
                self._clear_glm_recurrent_transients(entry)
        draft_cache.clear()
        target_cache.clear()


def _load_glm5_target_bundle(
    model_ref: str | Path | None,
    *,
    lazy: bool = True,
    quantize_kv_cache: bool = False,
    verify_config: Any | None = None,
) -> Any:
    """Load GLM through mlx-vlm while returning dflash-mlx's bundle type.

    The pinned DFlash loader requests lazy targets by default. GLM must follow
    the regular VLMEngine contract instead: mlx-vlm constructs the fallback
    target eagerly, then oMLX performs its bounded whole-tree materialization.
    Both routes run inside VLMEngine's pre-quantization sanitize scope; custom
    oQ loading otherwise owns its existing load policy unchanged.
    """
    del lazy, verify_config
    if quantize_kv_cache:
        raise ValueError("GLM-5.3 DFlash target KV quantization is unproven")
    if model_ref is None:
        raise ValueError("target model reference is required")

    from dflash_mlx.engine.target_ops import resolve_target_ops
    from dflash_mlx.runtime.loading import LoadedTargetBundle
    from mlx_vlm.utils import load as vlm_load

    from ..engine.vlm import _force_qwen4_exp_sanitize_on_load
    from ..utils.model_loading import (
        materialize_lazy_state,
        maybe_load_custom_quantization,
    )
    from .mlx_vlm_glm5_next_compat import apply_mlx_vlm_glm5_next_compat_patch

    apply_mlx_vlm_glm5_next_compat_patch()
    with _force_qwen4_exp_sanitize_on_load(Path(model_ref)):
        custom_loaded = maybe_load_custom_quantization(str(model_ref), is_vlm=True)
        if custom_loaded is not None:
            model, processor = custom_loaded
        else:
            model, processor = vlm_load(str(model_ref), lazy=False, strict=True)
    # The custom oQ loader intentionally leaves the target tree lazy. Normal
    # VLMEngine startup materializes it before serving, but DFlash owns a
    # separate loader and previously skipped that lifecycle step. The result
    # looked like a 1-second/~200 MB load and every target verification paid
    # the mmap/lazy-transform cost again. Force the exact same bounded model
    # materialization here, on DFlash's loader/MLX-executor thread, before any
    # speculative hook or request can observe the model.
    materialize_lazy_state(model)
    target_ops = resolve_target_ops(model)
    target_ops.install_speculative_hooks(model)
    config_path = Path(model_ref) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer = getattr(processor, "tokenizer", processor)
    return LoadedTargetBundle(
        model=model,
        tokenizer=tokenizer,
        meta={
            "resolved_model_ref": str(model_ref),
            "config": config,
            "quantize_kv_cache": False,
            "target_family": target_ops.family(model),
            "verify_linear_enabled": False,
            "verify_mode": "disabled-glm5-parity-gate",
        },
        target_ops=target_ops,
    )


def install_dflash_glm5_backend() -> bool:
    """Register GLM target ops and a scoped mlx-vlm target loader."""
    global _ORIGINAL_TARGET_LOADER

    from dflash_mlx.engine import target_ops
    from dflash_mlx.runtime import loading

    changed = False
    changed |= _install_glm5_prefill_chunking_bridge()
    try:
        from omlx.patches.dflash_glm5_prefix_cache import (
            install_glm5_dflash_prefix_cache,
        )

        changed |= install_glm5_dflash_prefix_cache()
    except Exception:
        import logging

        logging.getLogger(__name__).exception(
            "GLM DFlash prefix-cache enable failed; serving without it"
        )
    if _BACKEND_PATH not in target_ops.TARGET_BACKENDS:
        target_ops.TARGET_BACKENDS.append(_BACKEND_PATH)
        changed = True

    current = loading.load_target_bundle
    if not getattr(current, "_omlx_glm5_target_loader", False):
        _ORIGINAL_TARGET_LOADER = current

        def load_target_bundle(model_ref=None, **kwargs):
            candidate = Path(str(model_ref)).expanduser() if model_ref else None
            config_path = candidate / "config.json" if candidate is not None else None
            if config_path is None or not config_path.exists():
                return current(model_ref, **kwargs)
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return current(model_ref, **kwargs)
            if not _is_glm5_config(config):
                return current(model_ref, **kwargs)
            return _load_glm5_target_bundle(model_ref, **kwargs)

        load_target_bundle._omlx_glm5_target_loader = True  # type: ignore[attr-defined]
        load_target_bundle._omlx_original = current  # type: ignore[attr-defined]
        loading.load_target_bundle = load_target_bundle
        changed = True
    return changed


__all__ = [
    "Glm5NextTargetOps",
    "install_dflash_glm5_backend",
    "restore_glm5_dflash_class_patches",
    "validate_glm5_dflash_pair",
]
