# Memory System for Open WebUI

A long-term memory system that learns from conversations and personalizes responses without requiring external APIs or tokens.

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
Three specialized caches (embeddings, retrieval results, memory lookups) with LRU eviction keep responses fast while managing memory efficiently. Each user gets isolated cache storage.

**Real-Time Status Updates**  
Emits progress messages during operations: memory retrieval progress, consolidation status, operation summaries — keeping users informed without overwhelming them.

**Multilingual by Design**  
All prompts and logic work language-agnostically. Stores memories in English but processes any input language seamlessly.

## Model Support

**LLM Support**  
Tested with Gemini 2.5 Flash Lite, GPT-4o-mini, Qwen2.5-Instruct, and Mistral-Small. Should work with any model that supports structured outputs.

**Embedding Model Support**  
Supports any sentence-transformers model. The default `gte-multilingual-base` works well for diverse languages and is efficient enough for real-time use. Make sure to tweak thresholds if you switch to a different model.

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
- **model**: LLM for consolidation and reranking (default: `gemini-2.5-flash-lite`)
- **embedding_model**: Sentence transformer (default: `gte-multilingual-base`)
- **max_memories_returned**: Context injection limit (default: 10)
- **semantic_retrieval_threshold**: Minimum similarity score (default: 0.5)
- **enable_llm_reranking**: Toggle smart reranking (default: true)
- **llm_reranking_trigger_multiplier**: When to activate LLM (default: 0.5 = 50%)

## Performance Optimizations

- Batched embedding generation for efficiency
- Normalized embeddings for faster similarity computation
- Cached embeddings prevent redundant model calls
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