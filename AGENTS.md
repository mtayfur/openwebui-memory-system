# AGENTS.md

## Repo Snapshot

- Single-file OpenWebUI Filter extension: `memory_system.py`.
- No app, server, or CLI. Deployment is pasting the file into OpenWebUI Admin -> Functions.
- No tests or CI. Validation is formatting only.

## OpenWebUI Contracts

- `Filter` is the entry point.
- Keep `inlet(body, __event_emitter__, __user__, __request__)` and `outlet(body, __event_emitter__, __user__, __request__)` exact.
- OpenWebUI injects `__dunder__` args positionally. Renaming breaks integration.
- `inlet` injects memories into the system prompt and stores retrieval similarities in `RETRIEVAL_CACHE`.
- `outlet` must return immediately, reuse `RETRIEVAL_CACHE`, and launch consolidation with `asyncio.create_task()`.
- If `__user__` or `__request__` is missing, return `body`.
- `request.app.state.EMBEDDING_FUNCTION` is loaded lazily in `_initialize_system()`.
- Module docstring fields are machine-parsed. Keep `title: Memory System`, `version: 1.3.0`, and `required_open_webui_version: 0.6.40` accurate.

## Runtime-Only Imports

- `Users`, `Memories`, `generate_chat_completion`, and `Request` only resolve inside OpenWebUI.
- Standalone import of this module will fail.

## Key LLM Rules

- `_query_llm(..., response_model=...)` requires strict JSON-schema-capable models.
- `Prompts` f-strings are evaluated at class definition time. Changing `Constants` later does not update prompt text.
- `conversation_context` with more than one user message should be used for consolidation candidate selection, not just the last message.
- `memory_model=None` uses the current chat model for LLM operations.
- `llm_reranking_trigger_multiplier=0` disables LLM reranking.
- Higher `skip_category_margin` is more conservative.
- `status_emit_level` controls UI verbosity: `Basic`, `Intermediate`, or `Detailed`.
- `max_consolidation_context_messages` controls how many recent user messages are included in consolidation context.

## Cache and State

- `_SHARED_SKIP_DETECTOR_CACHE` is module-level and shared across `Filter` instances.
- `MEMORY_CACHE` can go stale after external OpenWebUI Memories edits. Refresh or restart to resync.
- `SkipDetector` category embeddings are computed once per embedding engine/model key.
- Semantic dedup on `UPDATE` can schedule a `DELETE` for the duplicate memory.
- If DELETE operations exceed 60% of total ops and there are at least 6 ops, the consolidation plan is rejected.
- Embeddings are stored as `np.float16`; similarity uses normalized dot products.

## Class Map

- `Filter`: OpenWebUI entry point, caches, valves, and background tasks.
- `Constants`: thresholds and limits.
- `Prompts`: prompt templates.
- `Models`: strict Pydantic response models.
- `UnifiedCacheManager`: global LRU for `embedding`, `retrieval`, and `memory`.
- `SkipDetector`: structural fast-path plus semantic classification.
- `LLMRerankingService`: selects relevant memories.
- `LLMConsolidationService`: collects candidates, builds plans, dedups, and executes ops.

## Dev Workflow

- Use `./dev-check.sh` for formatting.
- Manual format: `python -m black --line-length 160 memory_system.py` then `python -m isort memory_system.py`.
- Line length is 160.

## Tuning Knobs

- `SEMANTIC_RETRIEVAL_THRESHOLD` 0.20: lower = more recall, more noise.
- `DEDUPLICATION_SIMILARITY_THRESHOLD` 0.90: lower = more aggressive dedup.
- `SKIP_CATEGORY_MARGIN` 0.20: lower = more messages skipped.
- `MAX_DELETE_OPERATIONS_RATIO` 0.6: lower = more conservative.

## Skip Detector

- Fast-path structural patterns run before semantic comparison.
- Fast-path false positives are silent skips.

## Edit Rules

- Prefer the smallest correct change.
- Do not rename hook args, move runtime-only imports out of context, or change module metadata casually.
- If changing prompt text, remember `Constants` values are baked into `Prompts`.
