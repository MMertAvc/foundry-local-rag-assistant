"""Thin wrapper around the Foundry Local SDK.

Everything model-related is isolated here so the rest of the pipeline
(chunking, storage, retrieval) stays pure Python and unit-testable
without the SDK installed.

SDK reference (foundry-local-sdk >= 1.2):
    Configuration / FoundryLocalManager  -> runtime init
    catalog.get_model(alias).load()      -> download (first run) + load
    model.get_chat_client()              -> OpenAI-style chat completions
    model.get_embedding_client()         -> OpenAI-style embeddings
"""

from __future__ import annotations

import logging
from typing import List

from . import config

logger = logging.getLogger(__name__)


class FoundryEngine:
    """Manages the local chat + embedding models via Foundry Local."""

    def __init__(
        self,
        chat_alias: str = config.CHAT_MODEL_ALIAS,
        embed_alias: str = config.EMBED_MODEL_ALIAS,
    ) -> None:
        # Imported lazily so unit tests can run without the SDK.
        from pathlib import Path

        from foundry_local_sdk import Configuration, FoundryLocalManager

        # Reuse the `foundry` CLI's shared model cache instead of the
        # per-app_name default (~/.LocalRAGAssistant/cache/models). Without
        # this, the SDK looks in an empty app-specific directory and fails
        # with "Model path does not exist" even after `foundry model
        # download` has fetched the model into the shared cache.
        shared_cache_dir = str(Path.home() / ".foundry" / "cache" / "models")
        try:
            FoundryLocalManager.initialize(
                Configuration(app_name=config.APP_NAME, model_cache_dir=shared_cache_dir)
            )
        except Exception:
            # Singleton was already initialized elsewhere in this process.
            pass
        self._manager = FoundryLocalManager.instance
        catalog = self._manager.catalog

        self._embed_model = None  # set only when the Foundry backend is used
        self._embedder = self._create_embedder(catalog, embed_alias)
        self.embedder_name = self._embedder.name

        logger.info("Loading chat model '%s' ...", chat_alias)
        self._chat_model = catalog.get_model(chat_alias)
        self._prefer_cpu_variant(self._chat_model)
        self._chat_model.load()
        self._chat_client = self._chat_model.get_chat_client()
        self._chat_client.settings.temperature = config.TEMPERATURE
        self._chat_client.settings.max_tokens = config.MAX_TOKENS

    @staticmethod
    def _prefer_cpu_variant(model) -> None:
        """Force the CPU variant when the SDK's default pick can't run here.

        The SDK's default variant selection can pick a GPU/DirectML variant
        even on machines without a working DirectML execution provider,
        which fails to load with "DML provider requested, but the installed
        GenAI has not been built with DML support". CPU always works.
        """
        cpu_variant = next(
            (v for v in model.variants if str(v.info.runtime.device_type) == "CPU"),
            None,
        )
        if cpu_variant is not None:
            model.select_variant(cpu_variant)

    # --- Embeddings ---------------------------------------------------------

    def _create_embedder(self, catalog, embed_alias: str):
        """Pick the embedding backend per config.EMBEDDER_MODE.

        Foundry Local embedding models require Windows 11 24H2+ (build
        26100); on older builds the service excludes them from the
        catalog. In "auto" mode we detect that and fall back to a local
        ONNX embedder (fastembed) so the app stays 100% offline.
        """
        from .embedders import FastEmbedEmbedder, FoundryEmbedder

        mode = config.EMBEDDER_MODE.lower()
        if mode not in {"auto", "foundry", "local"}:
            raise ValueError(f"Invalid RAG_EMBEDDER={config.EMBEDDER_MODE!r}")

        if mode in {"auto", "foundry"}:
            try:
                logger.info("Loading Foundry embedding model '%s' ...", embed_alias)
                model = catalog.get_model(embed_alias)
                if model is None:
                    raise RuntimeError(
                        f"'{embed_alias}' not in the Foundry catalog on this "
                        "machine (embedding models need Windows 11 24H2+)."
                    )
                model.load()  # downloads on first run, then cached
                self._embed_model = model
                return FoundryEmbedder(model, embed_alias)
            except Exception as exc:
                if mode == "foundry":
                    raise
                logger.warning(
                    "Foundry embedding backend unavailable (%s). "
                    "Falling back to local ONNX embeddings (fastembed).", exc
                )

        return FastEmbedEmbedder(config.LOCAL_EMBED_MODEL)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Returns one vector per input text."""
        return self._embedder.embed_texts(texts)

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text (e.g. the user's query)."""
        return self.embed_texts([text])[0]

    # --- Chat ----------------------------------------------------------------

    def chat(self, system_prompt: str, user_message: str) -> str:
        """One-turn chat completion: system instructions + user question."""
        response = self._chat_client.complete_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        )
        return response.choices[0].message.content

    # --- Lifecycle -------------------------------------------------------------

    def close(self) -> None:
        """Unload models to free memory."""
        for model in (self._chat_model, self._embed_model):
            if model is None:
                continue
            try:
                model.unload()
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
