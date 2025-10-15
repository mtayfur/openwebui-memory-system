# Memory System for Open WebUI

A long-term memory system that learns from conversations and personalizes responses without requiring external APIs or tokens.

## Important Notice

**Privacy Consideration:** This system shares user messages and stored memories with your configured LLM for memory consolidation and retrieval operations. All data is processed through Open WebUI's built-in models using your existing configuration. No data is sent to external services beyond what your LLM provider configuration already allows.

## Core Features

**Zero External Dependencies**  
Uses Open WebUI's built-in models (LLM and embeddings) — no API keys, no external services.

**Intelligent Memory Consolidation**  
Automatically processes conversations in the background to create, update, or delete memories. The LLM analyzes context and decides when to store personal facts, enriching existing memories rather than creating duplicates.

**Hybrid Memory Retrieval**  
Starts with fast semantic search, then switches to LLM-powered reranking only when needed. The system triggers LLM reranking automatically when candidate count exceeds 50% of max retrieval limit, optimizing for both speed and accuracy.

**Smart Skip Detection**  
Avoids wasting resources on irrelevant messages through two-stage detection:
- **Fast-path**: Regex patterns catch technical content (code, logs, URLs, commands) instantly
- **Semantic**: Zero-shot classification identifies instructions, math, translations, and grammar requests

Categories automatically skipped: technical discussions, formatting requests, calculations, translation tasks, proofreading, and non-personal queries.

**Multi-Layer Caching**  
Three specialized caches (embeddings, retrieval, memory) with LRU eviction keep responses fast while managing memory efficiently. Each user gets isolated cache storage.

**Real-Time Status Updates**  
Emits progress messages during operations: memory retrieval progress, consolidation status, operation summaries — keeping users informed without overwhelming them.

**Multilingual by Design**  
All prompts and logic work language-agnostically. Stores memories in English but processes any input language seamlessly.

## Model Support

**LLM Support**  
Tested with gemini-2.5-flash-lite, gpt-5-nano, and qwen3-instruct. Should work with any model that supports structured outputs.

**Embedding Model Support**  
Uses OpenWebUI's configured embedding model (supports Ollama, OpenAI, Azure OpenAI, and local sentence-transformers). Configure embedding models through OpenWebUI's RAG settings. The memory system automatically uses whatever embedding backend you've configured in OpenWebUI.

## How It Works

**During Chat (Inlet)**  
1. Checks if message should be skipped (technical/instruction content)
2. Retrieves relevant memories using semantic search
3. Applies LLM reranking if candidate count is high
4. Injects top memories into context for personalized responses

**After Response (Outlet)**  
1. Runs consolidation in background without blocking
2. Gathers candidate memories using relaxed similarity threshold
3. LLM generates operations (CREATE/UPDATE/DELETE)
4. Executes validated operations and clears affected caches

## Configuration

Customize behavior through valves:
- **model**: LLM for consolidation and reranking (default: `google/gemini-2.5-flash-lite`)
- **max_message_chars**: Maximum message length before skipping operations (default: 2500)
- **max_memories_returned**: Context injection limit (default: 10)
- **semantic_retrieval_threshold**: Minimum similarity score (default: 0.5)
- **relaxed_semantic_threshold_multiplier**: Adjusts threshold for consolidation (default: 0.9)
- **enable_llm_reranking**: Toggle smart reranking (default: true)
- **llm_reranking_trigger_multiplier**: When to activate LLM reranking (default: 0.5 = 50%)

## Performance Optimizations

- Batched embedding generation for efficiency
- Normalized embeddings for faster similarity computation
- Cached embeddings prevent redundant API calls to OpenWebUI's embedding backend
- LRU eviction keeps memory footprint bounded
- Fast-path skip detection for instant filtering
- Selective LLM usage based on candidate count

## Memory Quality

The system maintains high-quality memories through:
- Temporal tracking with date anchoring
- Entity enrichment (combining names with descriptions)
- Relationship completeness (never stores partial connections)
- Contextual grouping (related facts stored together)
- Historical preservation (superseded facts converted to past tense)