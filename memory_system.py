"""
title: Memory System
description: A semantic memory management system for Open WebUI that consolidates, deduplicates, and retrieves personalized user memories using LLM operations.
version: 1.3.0
authors: https://github.com/mtayfur
license: Apache-2.0
required_open_webui_version: 0.6.40
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from fastapi import Request
from open_webui.models.users import Users
from open_webui.routers.memories import Memories
from open_webui.utils.chat import generate_chat_completion
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

logger = logging.getLogger(__name__)

MAX_SKIP_DETECTOR_CACHE_ENTRIES = 10
_SHARED_SKIP_DETECTOR_CACHE = OrderedDict()
_SHARED_SKIP_DETECTOR_CACHE_LOCK = asyncio.Lock()


class Constants:
    """Centralized configuration constants for the memory system."""

    # Core System Limits
    MAX_MEMORY_CONTENT_CHARS = 500  # Character limit for LLM prompt memory content
    MAX_MEMORY_SENTENCES = 5  # Sentence limit per memory for clarity
    MAX_MEMORIES_PER_RETRIEVAL = 10  # Maximum memories returned per query
    MAX_MESSAGE_CHARS = 5000  # Maximum message length for validation
    MIN_MESSAGE_CHARS = 10  # Minimum message length for validation
    MAX_CONSOLIDATION_CONTEXT_MESSAGES = 3  # Number of recent messages to include for pronoun/context resolution
    DATABASE_OPERATION_TIMEOUT_SEC = 10  # Timeout for DB operations like user lookup
    LLM_CONSOLIDATION_TIMEOUT_SEC = 60.0  # Timeout for LLM consolidation operations

    # Cache System
    MAX_CACHE_ENTRIES_PER_TYPE = 500  # Maximum cache entries per cache type
    MAX_CONCURRENT_USER_CACHES = 50  # Maximum concurrent user cache instances
    CACHE_KEY_HASH_PREFIX_LENGTH = 10  # Hash prefix length for cache keys

    # Retrieval & Similarity
    SEMANTIC_RETRIEVAL_THRESHOLD = 0.20  # Semantic similarity threshold for retrieval
    RELAXED_SEMANTIC_THRESHOLD_MULTIPLIER = 0.8  # Multiplier for relaxed similarity threshold in secondary operations
    EXTENDED_MAX_MEMORY_MULTIPLIER = 1.6  # Multiplier for expanding memory candidates in advanced operations
    LLM_RERANKING_TRIGGER_MULTIPLIER = 0.8  # Multiplier for LLM reranking trigger threshold

    # Skip Detection
    SKIP_CATEGORY_MARGIN = 0.20  # Margin above personal similarity for skip category classification
    DEDUPLICATION_SIMILARITY_THRESHOLD = 0.90  # Similarity threshold for deduplication checks

    # Safety & Operations
    MAX_DELETE_OPERATIONS_RATIO = 0.6  # Maximum delete operations ratio for safety
    MIN_OPS_FOR_DELETE_RATIO_CHECK = 6  # Minimum operations to apply ratio check

    # Content Display
    CONTENT_PREVIEW_LENGTH = 100  # Maximum length for content preview display

    # Status Emit Levels (maps enum string values to numeric levels for comparison)
    STATUS_LEVEL = {"Basic": 0, "Intermediate": 1, "Detailed": 2}


class Prompts:
    """Container for all LLM prompts used in the memory system."""

    MEMORY_CONSOLIDATION = f"""You are the Memory Consolidator. Extract and maintain precise personal facts from user messages as first-person English statements.

## INPUT
JSON object containing:
- `current_time`: Current UTC datetime for resolving relative dates.
- `user_message`: The latest user message (present when no conversation context is provided).
- `conversation`: Ordered array of recent user messages (user-authored only). When present, the LAST entry is the latest message. Analyze ALL entries for entity resolution.
- `existing_memories`: User's stored memories as `[id] content [noted at date]`. May be empty.

## OPERATIONS
- **CREATE**: `{{"operation": "CREATE", "id": "", "content": "..."}}` — New significant personal fact with lasting relevance.
- **UPDATE**: `{{"operation": "UPDATE", "id": "mem-xxx", "content": "..."}}` — Enrich an existing memory with newly revealed names, dates, or meaningful context. **Never UPDATE merely to rephrase, reword, or restructure a memory that is already accurate and complete.**
- **DELETE**: `{{"operation": "DELETE", "id": "mem-xxx", "content": ""}}` — Remove only when the user explicitly revokes a fact or states a direct contradiction.
- **SKIP**: Return `{{"ops": []}}` when no operation is warranted.

## DECISION SEQUENCE (evaluate in strict order — stop at first match)

### 1. Intent Gate
SKIP if the user's primary intent is NOT to state a personal fact. This includes:
- Instructional requests: rewrite, translate, proofread, debug, format, generate, summarize, compare, calculate, explain.
- Questions seeking help, advice, recommendations, or information — even when personal details appear as task parameters.
- Creative writing, persona roleplay, or content generation.
**Only proceed when the user is directly stating personal facts as part of their own narrative.**

### 2. Significance Gate
SKIP if the fact lacks lasting relevance. Exclude:
- Transient emotions, temporary states, momentary situations, one-time activities.
- General knowledge, casual mentions, or fleeting interests.
**Only proceed for enduring life facts: identity, relationships, career, residence, health, milestones, core preferences.**

### 3. Deduplication Gate
SKIP if an existing memory already captures the fact accurately and completely — even if worded differently.
**Proceed only if the message reveals genuinely new information (names, dates, context) that enriches or corrects an existing memory.**

### 4. Operation Selection
- **UPDATE preferred**: When an existing memory can be enriched with new details.
- **CREATE**: When no existing memory covers this fact. Decompose compound facts into separate self-contained memories grouped by subject.
- **Conservatism**: When uncertain whether something is a lasting fact or a transient mention, SKIP.

## CONTENT RULES
- **Language**: All memory content in English, first-person ("I", "My"), regardless of input language.
- **Entity Resolution**: Resolve ALL pronouns and generic references ("he", "she", "the project") to specific named entities using conversation context before storing. Always specify relationship context ("my colleague Maria", not just "Maria").
- **Temporal Precision**: Convert relative dates ("last month", "yesterday") to specific dates using `current_time`. Preserve history by transforming superseded facts into past-tense with defined time boundaries. Do not assign `current_time` as a start date for ongoing states without explicit temporal anchoring.
- **Conciseness**: Max {Constants.MAX_MEMORY_SENTENCES} sentences and {Constants.MAX_MEMORY_CONTENT_CHARS} characters per memory.
- **Cohesion**: Combine naturally related facts about the same subject (person, event, timeframe). Never merge unrelated information into one memory.

## EXAMPLES

### Example 1: Cohesive subject grouping
Message: "My wife Sarah loves hiking and outdoor activities. She has an active lifestyle and enjoys rock climbing. I started this new hobby last month and it's been great."
Memories: []
Return: {{"ops": [{{"operation": "CREATE", "id": "", "content": "My wife Sarah has an active lifestyle and enjoys hiking, outdoor activities, and rock climbing"}}, {{"operation": "CREATE", "id": "", "content": "I started rock climbing in August 2025 as a new hobby and have been enjoying it"}}]}}
Explanation: Sarah's interests are grouped as one cohesive fact about the same person. The user's own hobby is a separate subject.

### Example 2: Enriching existing memories with new details
Message: "My daughter Emma just turned 12. We adopted a dog named Max for her 11th birthday. What should I give her for her 12th birthday?"
Memories: [id:mem-002] My daughter Emma is 10 years old [noted at March 20 2024] [id:mem-101] I have a golden retriever [noted at September 20 2024]
Return: {{"ops": [{{"operation": "UPDATE", "id": "mem-002", "content": "My daughter Emma turned 12 years old in September 2025"}}, {{"operation": "UPDATE", "id": "mem-101", "content": "I have a golden retriever named Max that was adopted in September 2024 as a birthday gift for my daughter Emma when she turned 11"}}]}}
Explanation: Both memories enriched with new details (age update, dog's name, adoption context). Birthday gift question is instructional — ignored.

### Example 3: Relocation with history preservation
Message: "¿Me puedes recomendar buenos restaurantes de tapas en Barcelona? Me mudé aquí desde Madrid el mes pasado."
Memories: [id:mem-005] I live in Madrid Spain [noted at June 12 2025]
Return: {{"ops": [{{"operation": "UPDATE", "id": "mem-005", "content": "I lived in Madrid Spain until August 2025"}}, {{"operation": "CREATE", "id": "", "content": "I moved from Madrid to Barcelona Spain in August 2025"}}]}}
Explanation: Relocation is a significant life event — old location converted to past-tense. Restaurant request is instructional — ignored. Content in English despite Spanish input.

### Example 4: Contradiction triggers DELETE + CREATE
Message: "My wife Sofia and I just got married in August. What are some good honeymoon destinations?"
Memories: [id:mem-008] I am single [noted at January 5 2025]
Return: {{"ops": [{{"operation": "DELETE", "id": "mem-008", "content": ""}}, {{"operation": "CREATE", "id": "", "content": "I married Sofia in August 2025 and she is now my wife"}}]}}
Explanation: Marriage directly contradicts "single" status — DELETE the outdated fact. Honeymoon question is instructional — ignored.

### Example 5: Technical intent with personal data — SKIP
Message: "Fix this Python function that calculates my age from my birthdate of March 15, 1990."
Memories: []
Return: {{"ops": []}}
Explanation: Primary intent is technical. Birthdate appears as a task parameter, not a personal statement.

### Example 6: Accurate memory exists — no rephrase UPDATE
Message: "I work at Tesla as a software engineer."
Memories: [id:mem-010] I work as a software engineer at Tesla [noted at August 1 2025]
Return: {{"ops": []}}
Explanation: Existing memory captures the same fact accurately. Different word order is not new information.

### Example 7: Multi-message entity resolution
CONVERSATION:
[1] "I had lunch with my colleague Maria yesterday at the new cafe downtown."
[2] "She's been working on the same AI project as me for 3 months now."
[3] "She mentioned she might be leaving the company next month."
Memories: []
Return: {{"ops": [{{"operation": "CREATE", "id": "", "content": "My colleague Maria and I have been working together on an AI project for 3 months as of September 2025"}}, {{"operation": "CREATE", "id": "", "content": "My colleague Maria mentioned in September 2025 that she might be leaving the company in October 2025"}}]}}
Explanation: "She" in messages 2-3 resolved to "Maria" from message 1. Lunch location is transient — skipped. Work relationship and departure are significant facts.
"""

    MEMORY_RERANKING = f"""You are the Memory Relevance Selector. Choose which stored memories are relevant to personalizing the response to the user's current message.

## INPUT
JSON object containing:
- `current_time`: Current UTC datetime for assessing temporal relevance.
- `user_message`: The user's latest message.
- `candidate_memories`: Memories formatted as `[id] content [noted at date]`.

## OUTPUT
Return `{{"ids": [...]}}` — memory IDs ordered by relevance (most relevant first). Return `{{"ids": []}}` when no memories are relevant.

## RELEVANCE TIERS (select in this priority order)
1. **Direct**: Explicitly about the query topic, mentioned entities, or domain.
2. **Contextual**: Personal constraints that shape recommendations (preferences, budget, dietary needs, location, circumstances).
3. **Supporting**: Background facts useful only when they have a clear connection to the query topic.

## SELECTION RULES
1. **Temporal precedence**: When memories cover the same fact or contradict each other, prefer the later `noted at` date and exclude the outdated one. Include the older one only if the query concerns history or the transition itself.
2. **Connection required**: Every selected memory must have an identifiable link to the user's message. Do not include memories merely because they exist.
3. **Impersonal queries**: Return an empty list for purely technical or impersonal requests where no memory would improve the response.
4. **Limit**: At most {Constants.MAX_MEMORIES_PER_RETRIEVAL} IDs.

## EXAMPLES

### Example 1: Career history relevant to emotional context
Message: "I'm struggling with imposter syndrome at my new job. Any advice?"
Memories: [id:mem-001] I work as a senior software engineer at Tesla [noted at September 10 2025] [id:mem-002] I started my current job 3 months ago [noted at June 15 2025] [id:mem-003] I used to work in marketing [noted at March 5 2025] [id:mem-004] I graduated with a computer science degree [noted at May 15 2020]
Return: {{"ids": ["mem-001", "mem-002", "mem-003", "mem-004"]}}
Explanation: Career transition from marketing to software engineering directly informs imposter syndrome context.

### Example 2: Dietary and cuisine preferences for meal recommendation
Message: "Necesito ideas para una cena saludable y con muchas verduras esta noche."
Memories: [id:mem-030] I am trying a vegetarian diet [noted at September 1 2025] [id:mem-031] My favorite cuisine is Italian [noted at August 15 2025] [id:mem-032] I dislike spicy food [noted at August 5 2025]
Return: {{"ids": ["mem-030", "mem-031", "mem-032"]}}
Explanation: Vegetarian diet directly relevant. Cuisine preference and spice aversion shape personalized recommendations.

### Example 3: Interests and constraints for gift personalization
Message: "What are some good anniversary gift ideas for my wife, Sarah?"
Memories: [id:mem-101] My wife is named Sarah. [id:mem-102] My wife Sarah loves hiking and mystery novels. [id:mem-103] My wedding anniversary with Sarah is in October. [id:mem-104] I am on a tight budget this month. [id:mem-105] I live in Denver. [id:mem-106] I have a golden retriever named Max.
Return: {{"ids": ["mem-102", "mem-103", "mem-101", "mem-104"]}}
Explanation: Wife's interests and anniversary timing are direct. Budget is a contextual constraint. Location and pet have no connection to gift selection.

### Example 4: Impersonal technical query — empty list
Message: "I've been reading about quantum computing and I'm confused. Can you break down how quantum bits work differently from regular computer bits?"
Memories: [id:mem-026] I work as a senior software engineer at Tesla [noted at September 15 2025] [id:mem-027] My wife is named Sarah [noted at August 5 2025]
Return: {{"ids": []}}
Explanation: Purely technical explanation request. No memory would improve the response.
"""


class Models:
    """Container for all Pydantic models used in the memory system."""

    class StatusEmitLevel(str, Enum):
        """Verbosity levels for status message emission - selectable as dropdown with title case strings."""

        BASIC = "Basic"
        INTERMEDIATE = "Intermediate"
        DETAILED = "Detailed"

    class MemoryOperationType(Enum):
        CREATE = "CREATE"
        UPDATE = "UPDATE"
        DELETE = "DELETE"

    class StrictModel(BaseModel):
        """Base model with strict JSON schema for LLM structured output."""

        model_config = ConfigDict(extra="forbid")

    class MemoryOperation(StrictModel):
        """Pydantic model for memory operations with validation."""

        operation: "Models.MemoryOperationType" = Field(description="Type of memory operation to perform")
        content: str = Field(description="Memory content (required for CREATE/UPDATE, empty for DELETE)")
        id: str = Field(description="Memory ID (empty for CREATE, required for UPDATE/DELETE)")

        def validate_operation(self, existing_memory_ids: Optional[set] = None) -> bool:
            """Validate the memory operation against existing memory IDs."""
            if existing_memory_ids is None:
                existing_memory_ids = set()

            op_id = (self.id or "").strip()
            cleaned_content = (self.content or "").strip()

            if self.operation == Models.MemoryOperationType.CREATE:
                if cleaned_content:
                    self.content = cleaned_content
                    return True
                return False
            elif self.operation == Models.MemoryOperationType.UPDATE:
                if op_id and op_id in existing_memory_ids and cleaned_content:
                    self.id = op_id
                    self.content = cleaned_content
                    return True
                return False
            elif self.operation == Models.MemoryOperationType.DELETE:
                if op_id and op_id in existing_memory_ids:
                    self.id = op_id
                    return True
                return False
            return False

    class ConsolidationResponse(StrictModel):
        """Pydantic model for memory consolidation LLM response - object containing array of memory operations."""

        ops: List["Models.MemoryOperation"] = Field(default_factory=list, description="List of memory operations to execute")

    class MemoryRerankingResponse(StrictModel):
        """Pydantic model for memory reranking LLM response - object containing array of memory IDs."""

        ids: List[str] = Field(
            default_factory=list,
            description="List of memory IDs selected as most relevant for the user query",
        )


class UnifiedCacheManager:
    """Unified cache manager handling all cache types with global LRU eviction."""

    def __init__(self, max_cache_size_per_type: int, max_users: int):
        self.max_total_entries = max_cache_size_per_type * max_users
        self.caches: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.global_lru: OrderedDict[tuple, None] = OrderedDict()
        self._lock = asyncio.Lock()

        self.EMBEDDING_CACHE = "embedding"
        self.RETRIEVAL_CACHE = "retrieval"
        self.MEMORY_CACHE = "memory"

    async def get(self, user_id: str, cache_type: str, key: str) -> Optional[Any]:
        """Get value from cache with LRU updates."""
        async with self._lock:
            cache_key = (user_id, cache_type, key)
            if cache_key in self.global_lru:
                self.global_lru.move_to_end(cache_key)
                return self.caches[user_id][cache_type][key]
            return None

    async def put(self, user_id: str, cache_type: str, key: str, value: Any) -> None:
        """Store value in cache with global LRU eviction."""
        async with self._lock:
            cache_key = (user_id, cache_type, key)
            if cache_key in self.global_lru:
                self.global_lru.move_to_end(cache_key)
                self.caches[user_id][cache_type][key] = value
                return

            if len(self.global_lru) >= self.max_total_entries:
                oldest_key, _ = self.global_lru.popitem(last=False)
                o_user_id, o_cache_type, o_key = oldest_key
                del self.caches[o_user_id][o_cache_type][o_key]

                if not self.caches[o_user_id][o_cache_type]:
                    del self.caches[o_user_id][o_cache_type]
                if not self.caches[o_user_id]:
                    del self.caches[o_user_id]

            self.global_lru[cache_key] = None

            if user_id not in self.caches:
                self.caches[user_id] = {}
            if cache_type not in self.caches[user_id]:
                self.caches[user_id][cache_type] = {}

            self.caches[user_id][cache_type][key] = value

    async def remove(self, user_id: str, cache_type: str, key: str) -> bool:
        """Remove specific key from cache."""
        async with self._lock:
            cache_key = (user_id, cache_type, key)
            if cache_key in self.global_lru:
                del self.global_lru[cache_key]
                del self.caches[user_id][cache_type][key]

                if not self.caches[user_id][cache_type]:
                    del self.caches[user_id][cache_type]
                if not self.caches[user_id]:
                    del self.caches[user_id]
                return True
        return False

    async def clear_user_cache(self, user_id: str, cache_type: Optional[str] = None) -> int:
        """Clear specific cache type for user, or all caches for user if cache_type is None."""
        async with self._lock:
            if user_id not in self.caches:
                return 0

            keys_to_remove = []
            for k in self.global_lru.keys():
                if k[0] == user_id and (cache_type is None or k[1] == cache_type):
                    keys_to_remove.append(k)

            for k in keys_to_remove:
                del self.global_lru[k]
                del self.caches[k[0]][k[1]][k[2]]

                if not self.caches[k[0]][k[1]]:
                    del self.caches[k[0]][k[1]]
                if not self.caches[k[0]]:
                    del self.caches[k[0]]

            return len(keys_to_remove)

    async def clear_all_caches(self) -> None:
        """Clear all caches for all users."""
        async with self._lock:
            self.caches.clear()
            self.global_lru.clear()


class SkipDetector:
    """Binary content classifier: personal vs non-personal using semantic analysis."""

    NON_PERSONAL_CATEGORY_DESCRIPTIONS = [
        # --- Abstract Knowledge & Creative Tasks ---
        "General knowledge questions about impersonal, academic, or abstract topics like geography, world history, trivia, theoretical science, definitions, or factual information about the world.",
        "Explanations of concepts, mechanisms, or processes like photosynthesis, combustion engines, blockchain technology, DNA replication, the theory of relativity, or how things work in general.",
        "Creative writing prompts, requests for jokes, poems, fictional stories, songs, character backstories, marketing copy, or any form of content generation and creative output.",
        "Requests for generic recommendations, lists, outlines, or brainstorming on impersonal topics without personal context like movie ideas, company names, or essay structures.",
        "Requests for advice, suggestions, or recommendations where the primary intent is to get help or information, even if personal context is mentioned as part of the question.",
        "Seeking recommendations or help with personal decisions where the question is the focus, not stating facts, such as choosing destinations, products, jobs, or comparing options.",
        "Requests to generate, create, or design images, logos, graphics, illustrations, banners, visual content, or any form of media generation and visual design output.",
        "Requests to summarize, condense, extract key points, provide TL;DR, or distill the main takeaways from articles, documents, long text, or any form of content.",
        # --- Technical: Code & Programming ---
        "Programming language syntax, data types, algorithm logic, functions, methods, classes, object-oriented concepts, variable scope, control flow, modules, packages, or frameworks.",
        "Software design patterns including creational patterns like singleton and factory, structural patterns like adapter and decorator, and behavioral patterns like observer and strategy.",
        "Error handling, exceptions, stack traces, debugging errors like TypeError, NullPointerException, IndexError, segmentation fault, syntax errors, memory leaks, or runtime errors.",
        "HTTP status codes like 404 Not Found, 500 Internal Server Error, 200 OK, API responses, timeouts, CORS issues, Bad Gateway, Service Unavailable, or Too Many Requests.",
        "API design, REST endpoints, GraphQL, HTTP methods like GET, POST, PUT, DELETE, request-response cycles, authentication tokens, JWT, OAuth, API keys, or query parameters.",
        "Requests to review, debug, fix, or analyze code, find bugs in snippets, explain why code is not working, or identify issues with implementations and suggest corrections.",
        # --- Technical: DevOps, CLI & Data ---
        "Terminal commands, shell prompts, bash, zsh, powershell, filesystem navigation with cd, ls, pwd, file management with mkdir, rm, cp, mv, chmod, or text processing tools.",
        "Developer CLI tools, package managers, network requests with curl or wget, SSH access, version control with git commands, or containerization with docker and compose.",
        "Data interchange formats like JSON, XML, YAML, TOML, CSV, serialization, deserialization, parsing, config files, environment variables, or protocol buffers and schemas.",
        "WebSocket connections, real-time communication, socket programming, TCP, UDP, server-client connections, handshakes, HTTP upgrades, streaming, or pub-sub messaging patterns.",
        "File system paths, directory structures, absolute versus relative paths, operating systems, mount points, symbolic links, inodes, or file permissions with read, write, execute.",
        "Container orchestration, cluster management, service scaling, load balancing, Kubernetes pods, deployments, namespaces, Docker Swarm, container runtimes, or image registries.",
        "Database queries, SQL statements, tables, columns, rows, indexes, primary and foreign keys, joins, filters, relational versus NoSQL databases like MongoDB, PostgreSQL, or Redis.",
        "Application logging, log levels like INFO, WARN, ERROR, DEBUG, stack traces, timestamps, diagnostic telemetry, monitoring, or observability for system health and debugging.",
        # --- Technical: Algorithms & Testing ---
        "Algorithm analysis, time complexity like O(log n), space complexity, data structures including hash tables, arrays, linked lists, queues, stacks, heaps, graphs, or search algorithms.",
        "Sorting algorithms like merge sort, quicksort, insertion sort, selection sort, understanding stable versus unstable sorts, in-place operations, or computational complexity analysis.",
        "Regex patterns, regular expression matching, capturing groups, metacharacters, wildcards, quantifiers, character classes, lookaheads, lookbehinds, anchors, or word boundaries.",
        "Software testing, unit tests, assertions, mocks, stubs, fixtures, test suites, test cases, automated QA, testing frameworks like JUnit, pytest, Jest, or end-to-end testing.",
        "Cloud computing platforms like AWS, Azure, GCP, infrastructure as a service, compute instances, regions, availability zones, virtual machines, serverless functions, or CDN services.",
        "Markdown syntax for text formatting, headings, code blocks, inline code, emphasis with bold and italic, blockquotes, lists, task lists, tables, or horizontal rules and separators.",
        "Code formatting and style, indentation, tabs versus spaces, syntax highlighting for Python, JavaScript, Java, C++, Go, Rust, TypeScript, or linting tools like Prettier and ESLint.",
        # --- Instructional: Formatting & Rewriting ---
        "Formatting output as structured data, returning answers as JSON with specific keys, YAML, CSV, database-style tables with columns and rows, or lists of objects and arrays.",
        "Styling text presentation with markdown formatting like bullet points, numbered lists, task lists, tabular layouts with alignment, or hierarchical structures with nested elements.",
        "Adjusting response length to be shorter, more concise, brief, or condensed, summarizing key points, trimming text to reduce word count, or meeting specific character limits.",
        "Changing explanation depth to be more detailed, comprehensive, elaborate, thorough, or in-depth, expanding on points, or explaining topics with more complexity and nuance.",
        "Rewriting previous responses by rephrasing, paraphrasing, or reformulating with different wording, restating information in another way, or expressing meaning with new structure.",
        "Altering response tone to be more formal, academic, professional, casual, friendly, or conversational, adapting register and voice to suit a specific audience or context.",
        "Explaining concepts in simpler terms, breaking down topics step-by-step for beginners, clarifying confusing points, using analogies, or providing concrete examples for clarity.",
        "Continuing generated responses, keeping going with explanations or lists, providing more information, finishing thoughts, completing content, or proceeding with next steps.",
        "Acting as a specific persona or role like a pirate, scientist, or travel guide, adopting a character's voice, style, and knowledge base, or maintaining a persona throughout.",
        "Comparing and contrasting two or more topics, explaining similarities and differences between options, providing detailed analysis, or creating tables to highlight distinctions.",
        # --- Instructional: Math & Calculation ---
        "Performing arithmetic calculations with explicit numbers, solving expressions with multiply, add, subtract, divide, or computing numeric results following order of operations.",
        "Evaluating mathematical expressions containing numbers and operators, solving numerical problems, computing final results, simplifying arithmetic, or showing calculation steps.",
        "Converting units between measurement systems with numeric values like kilometers to miles, fahrenheit to celsius, or feet to centimeters for distance, weight, volume, or temperature.",
        "Calculating percentages of numbers, determining prices after discounts, computing tips on bills, finding proportional values, or calculating sales tax, interest, or ratios.",
        "Solving algebraic equations for variables like x, using quadratic formulas for numeric values, solving simultaneous linear equations, or isolating unknown variables.",
        "Performing geometry calculations with numeric measurements like area of circles, volume of cubes, circumference, perimeter, diameter, or computing square roots and powers.",
        "Calculating compound interest on investments or savings, computing future values, monthly mortgage payments, financial calculations, ROI, APR, or loan amortization schedules.",
        "Computing descriptive statistics for datasets of numbers like mean, median, mode, average, standard deviation, variance, range, quartiles, or percentiles for distributions.",
        "Calculating health and fitness metrics using numeric formulas like Body Mass Index from weight and height, basal metabolic rate, target heart rate, or caloric needs.",
        "Calculating time differences between two dates, finding how many days, hours, or minutes between points in time, elapsed duration, age from birthdays, or time until events.",
        # --- Instructional: Translation ---
        "Translating explicitly quoted text to foreign languages like Spanish, French, German, or other target languages, converting source text in quotes for direct language conversion.",
        "Asking how to say specific words or phrases in another language like thank you, computer, or goodbye in Japanese, Chinese, Korean, or requesting direct translations of terms.",
        "Converting blocks of text or paragraphs from source to target languages, translating content to Italian, Arabic, Portuguese, Russian, or performing language conversion requests.",
        "Providing translations for sentences into specific foreign languages like Turkish, Hindi, Polish, translating complete sentences often enclosed in quotes or brackets for clarity.",
        "Asking for translations of source text into target languages, requesting translated output in German, French, Dutch, or querying for the translated equivalent of provided text.",
        "Translating passages to specified foreign languages using direct command format, indicating clear source and target languages, or converting provided text content to new languages.",
        "Asking for foreign language words for common vocabulary like house, beautiful, water, providing single-word translations in Italian, Swedish, or other languages for basic terms.",
        "Asking how to say specific phrases in other languages, converting English phrases to equivalents in German or other languages for practical conversational or professional use.",
        "Translating informal or slang expressions to colloquial equivalents in target languages, capturing correct tone and nuance of informal language in casual conversational contexts.",
        "Providing formal and professional translations for business phrases, translating corporate email content to French or German with appropriate terminology and register for context.",
        # --- Instructional: Proofreading & Editing ---
        "Proofreading, reviewing, revising, or editing provided text for errors, checking drafts for typos and mistakes, correcting grammar, spelling, punctuation, or improving clarity.",
        "Proofreading for coherence, readability, or professionalism, polishing text to sound professional and error-free, checking textual quality, sentence structure, or effectiveness.",
        "Correcting grammatical issues like subject-verb agreement, verb tense, pronoun reference errors, misplaced modifiers, faulty sentence structure, or validating grammar correctness.",
        "Fixing passive voice, run-on sentences, comma splices, or sentence fragments, addressing punctuation errors with apostrophes, quotation marks, semicolons, or capitalization.",
        "Improving writing quality by suggesting better word choice, alternative phrasing, synonyms, refined expression, enhanced vocabulary, or restructured sentences for coherence.",
        "Removing wordiness, filler words, or redundancy from text, improving logical progression of ideas, eliminating awkward phrasing, or making writing flow better and connect ideas.",
        "Rewriting, rephrasing, paraphrasing, or reformulating text using different wording, restating information in another way, or expressing same meaning with new structure.",
        "Adapting writing tone to be more casual, friendly, or conversational, changing register and voice to suit specific audiences, or adjusting style while maintaining core message.",
        # --- Requests involving personal items (formatting/organizing) ---
        "Requests to format, organize, or structure personal information like my resume, my schedule, my task list, my grocery list, my travel itinerary, my budget, or my to-do items.",
        "Asking to create tables, lists, or schedules for personal data like format my hobbies as a list, organize my appointments, structure my plans, or present my information as bullet points.",
        "Format requests where personal items are listed as content to be formatted, like my favorite hobbies are reading and hiking format this as a list, or my skills include Python and SQL make a table.",
        "Requests to format lists of favorite things as bullet points or numbered items, like can you format my hobbies as a list, or put my interests into a table, structuring personal preferences.",
        "Help me write, draft, or compose personal communications like emails to my boss, messages to my landlord, notes for my family, or texts to my friends without memorizing the content.",
        "Requests to analyze, calculate, or compute values related to my personal situation like my savings rate, my commute distance, my BMI, my age, or my expenses without storing facts.",
        "Story problems or word problems with personal context like my commute is X miles, I ran X miles, splitting costs with friends, calculating how much I owe, or figuring personal totals.",
        "Requests to proofread, review, or edit personal documents like my email to my boss, my cover letter, my message to my landlord, or my text to a friend for grammar or clarity.",
        "Asking for help with personal decisions or recommendations involving personal context where the request for help is the focus, not stating biographical facts for memory.",
        # --- Transient States & Momentary Situations ---
        "Describing current temporary emotional states, fleeting feelings, or momentary moods without lasting significance like feeling stressed, tired, excited, frustrated, or happy today.",
        "Temporary emotions or passing states that are not enduring personal facts like being angry about a situation, nervous about tomorrow, or worried right now as transient feelings.",
        "Mentioning one-time events, temporary situations, or transient circumstances without lasting impact like having a presentation Friday, being at the store, or working late tonight.",
        "Describing momentary situations, current locations, or immediate activities like being in a meeting, driving to work, cooking dinner, or watching a movie as temporary circumstances.",
        # --- Instructional requests with personal context ---
        "Requests to use analogies, metaphors, or storytelling techniques to explain topics, like explain using an analogy, use metaphors to make it relatable, or add storytelling elements.",
        "Requests to simplify explanations for specific audiences like explain like I'm 5 years old, explain for someone without technical background, or make it simple for a beginner.",
        "Math questions mentioning family members as context like I'm arguing with my brother what's X, helping my son with homework, or questions involving people I know.",
        "Translating idioms, proverbs, or figurative expressions between languages like how do you say break a leg in French, what's the equivalent of early bird catches the worm.",
    ]

    PERSONAL_CATEGORY_DESCRIPTIONS = [
        "Statements regarding my name, birthdate, age, nationality, ethnicity, personality, beliefs, values, religion, culture, education history, degrees, or formative personal experiences.",
        "Details about my medical diagnoses, conditions, surgeries, medications, allergies, diets, physical measurements, fitness routines, sleep patterns, mental health, or wellness practices.",
        "Information about my family members, names, ages, relationships, occupations, health, or details about my spouse, partner, children, friends, or romantic relationship status.",
        "Facts about my pets, names and breeds, social activities, community involvement, or details about people in my life and interactions with my social circle and broader community.",
        "Facts about my job titles, employers, workplace, industry, career transitions, certifications, skills, colleagues, work arrangements, professional development, or career milestones.",
        "Details regarding my financial situation, income level, budget, investments, savings goals, debts, tax situations, legal matters, or financial obligations and commitments.",
        "Facts about my residence type, living arrangements, roommates, neighborhood, city, country, relocations, commute details, vehicles I own, or transportation methods I use.",
        "Facts about my past life events, milestones like graduations, marriages, or births, memorable experiences, achievements, formative moments, travel history, or biographical info.",
        "Descriptions of my enduring emotional states, attitudes toward people, deep-seated preferences, aversions, motivations, sources of stress or joy, or persistent feelings.",
        "Personal narratives about trying to improve myself like I'm trying to save money, I'm working on quitting smoking, I'm learning a new language, or ongoing self-development efforts.",
        "Details about my hobbies, recreation, creative pursuits, sports I play or watch, media preferences like favorite movies, books, music, games, or entertainment I enjoy.",
        "Stating that my life, my relationship, my work, or my emotional state feels like something abstract, such as my life feels like chaos, my relationship feels broken, my day was like a disaster.",
        "Asking to recall or remember personal biographical facts I shared earlier, like what's my wife's name, where did I say I work, remind me what I said about my hobbies or family.",
    ]

    class SkipReason(Enum):
        SKIP_SIZE = "SKIP_SIZE"
        SKIP_NON_PERSONAL = "SKIP_NON_PERSONAL"
        SKIP_ALL_NON_PERSONAL = "SKIP_ALL_NON_PERSONAL"

    # Inlet (retrieval) status messages
    INLET_STATUS_MESSAGES = {
        SkipReason.SKIP_SIZE: "📏 Message Length Out of Limits, Skipping Memory Retrieval",
        SkipReason.SKIP_NON_PERSONAL: "🚫 No Personal Content, Skipping Memory Retrieval",
    }

    # Outlet (consolidation) status messages
    OUTLET_STATUS_MESSAGES = {
        SkipReason.SKIP_ALL_NON_PERSONAL: "🚫 No Personal Content in Context, Skipping Memory Consolidation",
    }

    def __init__(self, embedding_function: Callable[[Union[str, List[str]]], Any]):
        """Initialize the skip detector with an embedding function and compute reference embeddings."""
        self.embedding_function = embedding_function
        self._reference_embeddings = None

    async def initialize(self) -> None:
        """Compute and cache embeddings for category descriptions."""
        if self._reference_embeddings is not None:
            return

        non_personal_embeddings = await self.embedding_function(self.NON_PERSONAL_CATEGORY_DESCRIPTIONS)
        personal_embeddings = await self.embedding_function(self.PERSONAL_CATEGORY_DESCRIPTIONS)

        self._reference_embeddings = {
            "non_personal": np.array(non_personal_embeddings),
            "personal": np.array(personal_embeddings),
        }

        logger.info(
            f"✅ SkipDetector initialized with {len(self.NON_PERSONAL_CATEGORY_DESCRIPTIONS)} non-personal and {len(self.PERSONAL_CATEGORY_DESCRIPTIONS)} personal categories"
        )

    def validate_message_size(self, message: str, max_message_chars: int) -> Optional[str]:
        """Validate message size constraints."""
        if not message or not message.strip():
            return SkipDetector.SkipReason.SKIP_SIZE.value
        trimmed = message.strip()
        if len(trimmed) < Constants.MIN_MESSAGE_CHARS or len(trimmed) > max_message_chars:
            return SkipDetector.SkipReason.SKIP_SIZE.value
        return None

    def _fast_path_skip_detection(self, message: str) -> Optional[bool]:
        """Language-agnostic structural pattern detection with high confidence and low false positive rate."""
        msg_len = len(message)
        if msg_len == 0:
            return None

        # Pre-compute line structures used by multiple patterns
        lines = message.split("\n")
        line_count = len(lines)
        non_empty_lines = [line for line in lines if line.strip()]
        non_empty_count = len(non_empty_lines)

        # Pattern 1: Multiple URLs (5+ full URLs indicates link lists or technical references)
        url_pattern_count = message.count("http://") + message.count("https://")
        if url_pattern_count >= 5:
            return True

        # Pattern 2: Long unbroken alphanumeric strings (tokens, hashes, base64)
        for word in message.split():
            cleaned = word.strip('.,;:!?()[]{}"\'"')
            if len(cleaned) > 80 and cleaned.replace("-", "").replace("_", "").isalnum():
                return True

        # Pattern 3: Markdown/text separators (repeated ---, ===, ___, ***)
        separator_patterns = ["---", "===", "___", "***"]
        for pattern in separator_patterns:
            standalone_count = sum(1 for line in lines if line.strip() == pattern)
            if standalone_count >= 4:
                return True

        # Pattern 4: Command-line patterns with context-aware detection
        if non_empty_lines:
            actual_command_lines = 0
            for line in non_empty_lines:
                stripped = line.strip()
                if stripped.startswith("$ ") and len(stripped) > 2:
                    parts = stripped[2:].split()
                    if parts and parts[0].isalnum():
                        actual_command_lines += 1
                elif "$ " in stripped:
                    dollar_index = stripped.find("$ ")
                    if dollar_index > 0 and stripped[dollar_index - 1] in (
                        " ",
                        ":",
                        "\t",
                    ):
                        parts = stripped[dollar_index + 2 :].split()
                        if parts and len(parts[0]) > 0 and (parts[0].isalnum() or parts[0] in ["curl", "wget", "git", "npm", "pip", "docker"]):
                            actual_command_lines += 1
                elif stripped.startswith("# ") and len(stripped) > 2:
                    rest = stripped[2:].strip()
                    if rest and not rest[0].isupper() and " " in rest:
                        actual_command_lines += 1
                elif stripped.startswith("> ") and len(stripped) > 2:
                    continue

            if actual_command_lines >= 1 and any(c in message for c in ["http://", "https://", " | "]):
                return True
            if actual_command_lines >= 3:
                return True

        # Pattern 5: High path/URL density (dots and slashes suggesting file paths or URLs)
        if msg_len > 30:
            slash_count = message.count("/") + message.count("\\")
            dot_count = message.count(".")
            path_chars = slash_count + dot_count
            if path_chars > 10 and (path_chars / msg_len) > 0.15:
                return True

        # Pattern 6: Markup character density (structured data)
        markup_chars = sum(message.count(c) for c in "{}[]<>")
        if markup_chars >= 6:
            if markup_chars / msg_len > 0.10:
                return True
            curly_count = message.count("{") + message.count("}")
            if curly_count >= 10:
                return True

        # Pattern 7: Structured nested content with colons (key: value patterns)
        if line_count >= 8 and non_empty_count > 0:
            colon_lines = sum(1 for line in non_empty_lines if ":" in line and not line.strip().startswith("#"))
            indented_lines = sum(1 for line in non_empty_lines if line.startswith((" ", "\t")))

            if colon_lines / non_empty_count > 0.4 and indented_lines / non_empty_count > 0.5:
                words_outside_kv = 0
                for line in non_empty_lines:
                    if ":" not in line:
                        words_outside_kv += len(line.split())

                if words_outside_kv < 5:
                    return True

        # Pattern 8: Highly structured multi-line content (require markup chars for technical confidence)
        if line_count > 15 and non_empty_count > 0:
            markup_in_lines = sum(1 for line in non_empty_lines if any(c in line for c in "{}[]<>"))
            structured_lines = sum(1 for line in non_empty_lines if line.startswith((" ", "\t")))

            if markup_in_lines / non_empty_count > 0.3:
                return True
            elif structured_lines / non_empty_count > 0.6:
                operators = ["=", "+", "-", "*", "/", "<", ">", "&", "|", "!", ":", "?"]
                operator_count = sum(message.count(op) for op in operators)
                if (operator_count / msg_len) > 0.05:
                    return True

        # Pattern 9: Code-like indentation pattern (require code indicators to avoid false positives from bullet lists)
        if line_count >= 3 and non_empty_count > 0:
            indented_lines = sum(1 for line in non_empty_lines if line[0] in (" ", "\t"))
            if indented_lines / non_empty_count > 0.5:
                code_ending_chars = ["{", "}", "(", ")", ";"]
                lines_with_code_endings = sum(1 for line in non_empty_lines if line.strip().endswith(tuple(code_ending_chars)))
                if lines_with_code_endings / non_empty_count > 0.2:
                    return True

        # Pattern 10: Very high special character ratio (encoded data, technical output)
        if msg_len > 50:
            special_chars = sum(1 for c in message if not c.isalnum() and not c.isspace())
            special_ratio = special_chars / msg_len
            if special_ratio > 0.35:
                alphanumeric = sum(1 for c in message if c.isalnum())
                if alphanumeric / msg_len < 0.50:
                    return True

        return None

    async def detect_skip_reason(
        self,
        message: str,
        max_message_chars: int,
        memory_system: "Filter",
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Detect if a message should be skipped using two-stage detection: fast-path structural patterns and binary semantic classification."""
        size_issue = self.validate_message_size(message, max_message_chars)
        if size_issue:
            return size_issue

        fast_skip = self._fast_path_skip_detection(message)
        if fast_skip:
            logger.info(f"⚡ Fast-path skip: {self.SkipReason.SKIP_NON_PERSONAL.value}")
            return self.SkipReason.SKIP_NON_PERSONAL.value

        if self._reference_embeddings is None:
            await self.initialize()

        # Use memory_system's embedding generation to leverage per-user caching if possible
        if user_id:
            message_embedding = await memory_system._generate_embeddings(message.strip(), user_id)
        else:
            message_embedding_result = await self.embedding_function([message.strip()])
            message_embedding = np.array(message_embedding_result[0])

        personal_similarities = np.dot(message_embedding, self._reference_embeddings["personal"].T)
        max_personal_similarity = personal_similarities.max()

        non_personal_similarities = np.dot(message_embedding, self._reference_embeddings["non_personal"].T)
        max_non_personal_similarity = non_personal_similarities.max()

        margin = memory_system.valves.skip_category_margin
        threshold = max_personal_similarity + margin
        if (max_non_personal_similarity - max_personal_similarity) > margin:
            logger.info(f"🚫 Skipping: non-personal content (sim {max_non_personal_similarity:.3f} > {threshold:.3f})")
            return self.SkipReason.SKIP_NON_PERSONAL.value

        logger.info(f"✅ Allowing: personal content (sim {max_non_personal_similarity:.3f} <= {threshold:.3f})")
        return None


class LLMRerankingService:
    """Language-agnostic LLM-based memory reranking service."""

    def __init__(self, memory_system):
        self.memory_system = memory_system

    def _should_use_llm_reranking(self, memories: List[Dict]) -> Tuple[bool, str]:
        """Determine if LLM reranking should be used based on candidate count."""
        if self.memory_system.valves.llm_reranking_trigger_multiplier <= 0:
            return False, "LLM reranking disabled"

        llm_trigger_threshold = int(self.memory_system.valves.max_memories_returned * self.memory_system.valves.llm_reranking_trigger_multiplier)
        if len(memories) > llm_trigger_threshold:
            return (
                True,
                f"{len(memories)} candidate memories exceed {llm_trigger_threshold} threshold",
            )

        return (
            False,
            f"{len(memories)} candidate memories within threshold of {llm_trigger_threshold}",
        )

    async def _llm_select_memories(
        self,
        user_message: str,
        candidate_memories: List[Dict],
        max_count: int,
        request: Request,
        user: Dict[str, Any],
        model: str,
    ) -> List[Dict]:
        """Use LLM to select most relevant memories."""
        memory_lines = self.memory_system._format_memories_for_llm(candidate_memories)

        user_prompt = json.dumps(
            {
                "current_time": self.memory_system.format_current_datetime(),
                "user_message": user_message,
                "candidate_memories": memory_lines,
            },
            indent=2,
        )

        response = await self.memory_system._query_llm(
            Prompts.MEMORY_RERANKING,
            user_prompt,
            request,
            user,
            model,
            response_model=Models.MemoryRerankingResponse,
        )

        memory_map = {m["id"]: m for m in candidate_memories}
        selected_memories = []
        seen_ids = set()
        for memory_id in response.ids:
            if memory_id in seen_ids:
                continue
            memory = memory_map.get(memory_id)
            if memory:
                selected_memories.append(memory)
                seen_ids.add(memory_id)
                if len(selected_memories) >= max_count:
                    break

        logger.info(f"🧠 LLM selected {len(selected_memories)} out of {len(candidate_memories)} candidates")

        return selected_memories

    async def rerank_memories(
        self,
        user_message: str,
        candidate_memories: List[Dict],
        request: Request,
        user: Dict[str, Any],
        model: str,
        emitter: Optional[Callable] = None,
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """Rerank candidate memories using LLM or return top semantic matches."""
        start_time = time.time()
        max_injection = self.memory_system.valves.max_memories_returned

        should_use_llm, decision_reason = self._should_use_llm_reranking(candidate_memories)

        analysis_info = {
            "llm_decision": should_use_llm,
            "decision_reason": decision_reason,
            "candidate_count": len(candidate_memories),
        }

        if should_use_llm:
            extended_count = int(self.memory_system.valves.max_memories_returned * Constants.EXTENDED_MAX_MEMORY_MULTIPLIER)
            llm_candidates = candidate_memories[:extended_count]
            await self.memory_system._emit_status(
                emitter,
                f"🤖 Analyzing {len(llm_candidates)} Memories for Relevance",
                done=False,
                level=Constants.STATUS_LEVEL["Intermediate"],
            )
            logger.info(f"🧠 Using LLM reranking: {decision_reason}")

            selected_memories = await self._llm_select_memories(user_message, llm_candidates, max_injection, request, user, model)

            if not selected_memories:
                logger.info("📭 No relevant memories after LLM analysis")
                await self.memory_system._emit_status(
                    emitter,
                    "📭 No Relevant Memories Found",
                    done=True,
                    level=Constants.STATUS_LEVEL["Intermediate"],
                )
                return selected_memories, analysis_info
        else:
            logger.info(f"⏩ Skipping LLM reranking: {decision_reason}")
            selected_memories = candidate_memories[:max_injection]

        duration = time.time() - start_time
        duration_text = f" in {duration:.2f}s" if duration >= 0.01 else ""
        retrieval_method = "LLM" if should_use_llm else "Semantic"
        await self.memory_system._emit_status(
            emitter,
            f"🎯 {retrieval_method} Memory Retrieval Complete{duration_text}",
            done=True,
            level=Constants.STATUS_LEVEL["Detailed"],
        )
        return selected_memories, analysis_info


class LLMConsolidationService:
    """Language-agnostic LLM-based memory consolidation service."""

    def __init__(self, memory_system):
        self.memory_system = memory_system

    async def _check_semantic_duplicate(
        self,
        content_embedding: np.ndarray,
        memory_embeddings: List[np.ndarray],
        memories: List,
        exclude_id: Optional[str] = None,
    ) -> Optional[str]:
        """Check if content embedding is semantically duplicate of existing memories."""
        for i, memory_embedding in enumerate(memory_embeddings):
            if memory_embedding is None:
                continue
            memory = memories[i]
            if exclude_id and memory.id == exclude_id:
                continue

            similarity = np.dot(content_embedding, memory_embedding)
            if similarity >= Constants.DEDUPLICATION_SIMILARITY_THRESHOLD:
                logger.info(f"🔍 Semantic duplicate detected: similarity={similarity:.3f} with memory {memory.id}")
                return memory.id

        return None

    def _filter_consolidation_candidates(self, similarities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
        """Filter consolidation candidates by threshold and return candidates with threshold info."""
        consolidation_threshold = self.memory_system._get_retrieval_threshold(is_consolidation=True)
        candidates = [mem for mem in similarities if mem["relevance"] >= consolidation_threshold]

        max_consolidation_memories = int(self.memory_system.valves.max_memories_returned * Constants.EXTENDED_MAX_MEMORY_MULTIPLIER)
        candidates = candidates[:max_consolidation_memories]

        threshold_info = f"{consolidation_threshold:.3f} (max: {max_consolidation_memories})"
        return candidates, threshold_info

    async def collect_consolidation_candidates(
        self,
        user_message: str,
        user_id: str,
        cached_similarities: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Collect candidate memories for consolidation analysis using cached or computed similarities."""
        if cached_similarities:
            candidates, threshold_info = self._filter_consolidation_candidates(cached_similarities)

            logger.info(f"🎯 Found {len(candidates)} cached candidates for consolidation (threshold: {threshold_info})")

            self.memory_system._log_retrieved_memories(candidates, "consolidation")
            return candidates

        try:
            user_memories = await self.memory_system._get_user_memories(user_id)
        except asyncio.TimeoutError:
            raise TimeoutError(f"⏱️ Memory retrieval timed out after {Constants.DATABASE_OPERATION_TIMEOUT_SEC}s")

        if not user_memories:
            logger.info("💭 No existing memories found for consolidation")
            return []

        logger.info(f"🚀 Processing {len(user_memories)} cached memories for consolidation")

        _, all_similarities = await self.memory_system._compute_similarities(user_message, user_id, user_memories)

        if all_similarities:
            candidates, threshold_info = self._filter_consolidation_candidates(all_similarities)
        else:
            candidates = []
            threshold_info = "N/A"

        logger.info(f"🎯 Found {len(candidates)} candidates for consolidation (threshold: {threshold_info})")

        self.memory_system._log_retrieved_memories(candidates, "consolidation")

        return candidates

    async def generate_consolidation_plan(
        self,
        user_message: str,
        candidate_memories: List[Dict[str, Any]],
        request: Request,
        user: Dict[str, Any],
        model: str,
        emitter: Optional[Callable] = None,
        conversation_context: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate consolidation plan using LLM with clear system/user prompt separation."""
        memory_lines = self.memory_system._format_memories_for_llm(candidate_memories)

        prompt_data = {
            "current_time": self.memory_system.format_current_datetime(),
            "existing_memories": memory_lines,
        }

        if conversation_context and len(conversation_context) > 1:
            prompt_data["conversation"] = conversation_context
        else:
            prompt_data["user_message"] = user_message

        user_prompt = json.dumps(prompt_data, indent=2)

        response = await asyncio.wait_for(
            self.memory_system._query_llm(
                Prompts.MEMORY_CONSOLIDATION,
                user_prompt,
                request,
                user,
                model,
                response_model=Models.ConsolidationResponse,
            ),
            timeout=Constants.LLM_CONSOLIDATION_TIMEOUT_SEC,
        )

        operations = response.ops
        existing_memory_ids = {memory["id"] for memory in candidate_memories}

        total_operations = len(operations)
        delete_operations = [op for op in operations if op.operation == Models.MemoryOperationType.DELETE]
        delete_ratio = len(delete_operations) / total_operations if total_operations > 0 else 0

        if delete_ratio > Constants.MAX_DELETE_OPERATIONS_RATIO and total_operations >= Constants.MIN_OPS_FOR_DELETE_RATIO_CHECK:
            logger.warning(
                f"⚠️ Consolidation safety: {len(delete_operations)}/{total_operations} operations are deletions ({delete_ratio * 100:.1f}%) - rejecting plan"
            )
            return []

        deduplicated_operations = []
        seen_contents = set()
        seen_update_ids = set()

        for op in operations:
            if not op.validate_operation(existing_memory_ids):
                continue

            if op.operation == Models.MemoryOperationType.UPDATE and op.id in seen_update_ids:
                logger.info(f"⏭️ Skipping duplicate UPDATE for memory {op.id}")
                continue

            if op.operation in [
                Models.MemoryOperationType.CREATE,
                Models.MemoryOperationType.UPDATE,
            ]:
                normalized_content = op.content.strip().lower()
                if normalized_content in seen_contents:
                    op_type = "CREATE" if op.operation == Models.MemoryOperationType.CREATE else f"UPDATE {op.id}"
                    logger.info(f"⏭️ Skipping duplicate {op_type}: {self.memory_system._truncate_content(op.content)}")
                    continue
                seen_contents.add(normalized_content)

            if op.operation == Models.MemoryOperationType.UPDATE:
                seen_update_ids.add(op.id)
            deduplicated_operations.append(op.model_dump())

        valid_operations = deduplicated_operations

        if valid_operations:
            create_count = sum(1 for op in valid_operations if op.get("operation") == Models.MemoryOperationType.CREATE.value)
            update_count = sum(1 for op in valid_operations if op.get("operation") == Models.MemoryOperationType.UPDATE.value)
            delete_count = sum(1 for op in valid_operations if op.get("operation") == Models.MemoryOperationType.DELETE.value)

            operation_details = self.memory_system._build_operation_details(create_count, update_count, delete_count)

            logger.info(f"🎯 Planned {len(valid_operations)} operations: {', '.join(operation_details)}")
        else:
            logger.info("🎯 No valid operations planned")

        return valid_operations

    async def _deduplicate_operations(
        self,
        operations: List,
        valid_memories: List,
        memory_embeddings: List[np.ndarray],
        user_id: str,
        operation_type: str,
        delete_operations: Optional[List] = None,
    ) -> List:
        """Semantically deduplicate operations against existing memories. For UPDATEs, preserve enriched content and delete duplicates."""
        if not operations:
            return []

        deduplicated = []

        id_to_index = {m.id: i for i, m in enumerate(valid_memories)}

        op_contents = [op.content for op in operations]
        op_embeddings = await self.memory_system._generate_embeddings(op_contents, user_id)

        for i, operation in enumerate(operations):
            op_embedding = op_embeddings[i]
            if op_embedding is None or not valid_memories:
                deduplicated.append(operation)
                continue

            if operation_type == "UPDATE":
                target_idx = id_to_index.get(operation.id)
                if target_idx is not None and target_idx < len(memory_embeddings) and memory_embeddings[target_idx] is not None:
                    similarity = np.dot(op_embedding, memory_embeddings[target_idx])
                    if similarity >= Constants.DEDUPLICATION_SIMILARITY_THRESHOLD:
                        logger.info(f"⏭️ Skipping redundant UPDATE for {operation.id}: content similar ({similarity:.3f})")
                        continue

            exclude_id = operation.id if operation_type == "UPDATE" else None
            duplicate_id = await self._check_semantic_duplicate(op_embedding, memory_embeddings, valid_memories, exclude_id)

            if duplicate_id:
                if operation_type == "UPDATE" and delete_operations is not None:
                    logger.info(f"🔄 UPDATE creates duplicate: keeping {operation.id}, deleting {duplicate_id}")
                    deduplicated.append(operation)
                    delete_operations.append(
                        Models.MemoryOperation(
                            operation=Models.MemoryOperationType.DELETE,
                            content="",
                            id=duplicate_id,
                        )
                    )
                else:
                    logger.info(f"⏭️ Skipping duplicate {operation_type}: {self.memory_system._truncate_content(operation.content)} (matches {duplicate_id})")
                continue

            deduplicated.append(operation)

        return deduplicated

    async def execute_memory_operations(
        self,
        operations: List[Dict[str, Any]],
        user_id: str,
        emitter: Optional[Callable] = None,
    ) -> Tuple[int, int, int, int]:
        """Execute consolidation operations with simplified tracking."""
        if not operations:
            return 0, 0, 0, 0

        user = await asyncio.wait_for(
            Users.get_user_by_id(user_id),
            timeout=Constants.DATABASE_OPERATION_TIMEOUT_SEC,
        )

        created_count = updated_count = deleted_count = failed_count = 0

        operations_by_type = {"CREATE": [], "UPDATE": [], "DELETE": []}
        for operation_data in operations:
            try:
                operation = Models.MemoryOperation(**operation_data)
                operations_by_type[operation.operation.value].append(operation)
            except Exception as e:
                if isinstance(e, asyncio.CancelledError):
                    raise
                failed_count += 1
                operation_type = operation_data.get("operation", "UNSUPPORTED")
                content_preview = ""
                if "content" in operation_data:
                    content = operation_data.get("content", "")
                    content_preview = f" - Content: {self.memory_system._truncate_content(content, Constants.CONTENT_PREVIEW_LENGTH)}"
                elif "id" in operation_data:
                    content_preview = f" - ID: {operation_data['id']}"
                error_message = f"Failed {operation_type} operation{content_preview}: {str(e)}"
                logger.error(error_message)

        memory_cache_key = self.memory_system._cache_key(self.memory_system._cache_manager.MEMORY_CACHE, user_id)
        user_memories = await self.memory_system._cache_manager.get(user_id, self.memory_system._cache_manager.MEMORY_CACHE, memory_cache_key)

        if user_memories is None:
            user_memories = await self.memory_system._get_user_memories(user_id)
            await self.memory_system._cache_manager.put(
                user_id,
                self.memory_system._cache_manager.MEMORY_CACHE,
                memory_cache_key,
                user_memories,
            )

        memory_contents_for_deletion = {mem.id: mem.content for mem in user_memories} if (operations_by_type["DELETE"] or operations_by_type["UPDATE"]) else {}
        deleted_contents_for_cache = []

        # Optimization: Pre-compute valid memories and their embeddings once for all dedup operations
        valid_memories = [m for m in user_memories if m.content and len(m.content.strip()) >= Constants.MIN_MESSAGE_CHARS]
        memory_embeddings = []
        if valid_memories:
            memory_contents = [m.content for m in valid_memories]
            memory_embeddings = await self.memory_system._generate_embeddings(memory_contents, user_id)

        if operations_by_type["CREATE"]:
            operations_by_type["CREATE"] = await self._deduplicate_operations(
                operations_by_type["CREATE"],
                valid_memories,
                memory_embeddings,
                user_id,
                operation_type="CREATE",
            )

        if operations_by_type["UPDATE"]:
            operations_by_type["UPDATE"] = await self._deduplicate_operations(
                operations_by_type["UPDATE"],
                valid_memories,
                memory_embeddings,
                user_id,
                operation_type="UPDATE",
                delete_operations=operations_by_type["DELETE"],
            )

        for operation_type, ops in operations_by_type.items():
            if not ops:
                continue

            results = []
            for operation in ops:
                try:
                    result = await self.memory_system._execute_single_operation(operation, user)
                    results.append(result)
                except Exception as e:
                    results.append(e)

            for idx, result in enumerate(results):
                operation = ops[idx]

                if isinstance(result, Exception):
                    failed_count += 1
                    await self.memory_system._emit_status(
                        emitter,
                        f"❌ Failed {operation_type}",
                        done=False,
                        level=Constants.STATUS_LEVEL["Intermediate"],
                    )
                elif result == Models.MemoryOperationType.CREATE.value:
                    created_count += 1
                    content_preview = self.memory_system._truncate_content(operation.content)
                    await self.memory_system._emit_status(
                        emitter,
                        f"📝 Created: {content_preview}",
                        done=False,
                        level=Constants.STATUS_LEVEL["Intermediate"],
                    )
                elif result == Models.MemoryOperationType.UPDATE.value:
                    updated_count += 1
                    new_content_preview = self.memory_system._truncate_content(operation.content)
                    await self.memory_system._emit_status(
                        emitter,
                        f"✏️ Updated: {new_content_preview}",
                        done=False,
                        level=Constants.STATUS_LEVEL["Intermediate"],
                    )

                    old_content = memory_contents_for_deletion.get(operation.id)
                    if old_content:
                        deleted_contents_for_cache.append(old_content)

                elif result == Models.MemoryOperationType.DELETE.value:
                    deleted_count += 1
                    old_content = memory_contents_for_deletion.get(operation.id, operation.id)
                    if old_content and old_content != operation.id:
                        deleted_contents_for_cache.append(old_content)
                    old_content_preview = self.memory_system._truncate_content(old_content) if old_content else operation.id
                    await self.memory_system._emit_status(
                        emitter,
                        f"🗑️ Deleted: {old_content_preview}",
                        done=False,
                        level=Constants.STATUS_LEVEL["Intermediate"],
                    )

        total_executed = created_count + updated_count + deleted_count
        logger.info(
            f"✅ Memory processing completed: {total_executed}/{len(operations)} ops (created: {created_count}, updated: {updated_count}, deleted: {deleted_count}, failed: {failed_count})"
        )

        if total_executed > 0:
            operation_details = self.memory_system._build_operation_details(created_count, updated_count, deleted_count)
            logger.info(f"🔄 Memory operations: {', '.join(operation_details)}")
            await self.memory_system._refresh_user_cache(user_id, deleted_contents_for_cache)

        return created_count, updated_count, deleted_count, failed_count

    async def run_consolidation_pipeline(
        self,
        user_message: str,
        user_id: str,
        request: Request,
        user: Dict[str, Any],
        model: str,
        emitter: Optional[Callable] = None,
        cached_similarities: Optional[List[Dict[str, Any]]] = None,
        conversation_context: Optional[List[str]] = None,
    ) -> None:
        """Complete consolidation pipeline with simplified flow."""
        start_time = time.time()
        try:
            if self.memory_system._shutdown_event.is_set():
                return

            candidates = await self.collect_consolidation_candidates(user_message, user_id, cached_similarities)
            if self.memory_system._shutdown_event.is_set():
                return

            operations = await self.generate_consolidation_plan(
                user_message,
                candidates,
                request,
                user,
                model,
                emitter,
                conversation_context,
            )
            if self.memory_system._shutdown_event.is_set():
                return

            if operations:
                (
                    created_count,
                    updated_count,
                    deleted_count,
                    failed_count,
                ) = await self.execute_memory_operations(operations, user_id, emitter)

                duration = time.time() - start_time
                logger.info(f"💾 Memory consolidation complete in {duration:.2f}s")

                total_operations = created_count + updated_count + deleted_count
                if total_operations > 0 or failed_count > 0:
                    await self.memory_system._emit_status(
                        emitter,
                        f"💾 Memory Consolidation Complete in {duration:.2f}s",
                        done=False,
                        level=Constants.STATUS_LEVEL["Detailed"],
                    )

                    operation_details = self.memory_system._build_operation_details(created_count, updated_count, deleted_count)
                    memory_word = "Memory" if total_operations == 1 else "Memories"
                    operations_summary = f"{', '.join(operation_details)} {memory_word}"

                    if failed_count > 0:
                        operations_summary += f" (❌ {failed_count} Failed)"

                    await self.memory_system._emit_status(
                        emitter,
                        operations_summary,
                        done=True,
                        level=Constants.STATUS_LEVEL["Basic"],
                    )
            else:
                duration = time.time() - start_time
                await self.memory_system._emit_status(
                    emitter,
                    "✅ No Memory Updates Needed",
                    done=True,
                    level=Constants.STATUS_LEVEL["Detailed"],
                )

        except Exception as e:
            if isinstance(e, asyncio.CancelledError):
                raise
            duration = time.time() - start_time
            raise RuntimeError(f"❌ Memory consolidation failed after {duration:.2f}s: {str(e)}")


class Filter:
    """Enhanced multi-model embedding and memory filter with LRU caching."""

    class Valves(BaseModel):
        """Configuration valves for the Memory System."""

        memory_model: Optional[str] = Field(
            default=None,
            description="Custom model to use for memory operations. If Default, uses the current chat model",
        )
        max_memories_returned: int = Field(
            default=Constants.MAX_MEMORIES_PER_RETRIEVAL,
            description="Maximum number of memories to return in context",
        )
        semantic_retrieval_threshold: float = Field(
            default=Constants.SEMANTIC_RETRIEVAL_THRESHOLD,
            description="Minimum similarity threshold for memory retrieval",
        )
        llm_reranking_trigger_multiplier: float = Field(
            default=Constants.LLM_RERANKING_TRIGGER_MULTIPLIER,
            description="Controls when LLM reranking activates (0.0 = disabled, lower = more aggressive)",
        )
        skip_category_margin: float = Field(
            default=Constants.SKIP_CATEGORY_MARGIN,
            description="Margin above personal similarity for skip category classification (higher = more conservative skip detection)",
        )
        status_emit_level: Literal["Basic", "Intermediate", "Detailed"] = Field(
            default="Intermediate",
            description="Status message verbosity level: Basic (summary counts only), Intermediate (summaries and key details), Detailed (all details)",
        )
        max_consolidation_context_messages: int = Field(
            default=Constants.MAX_CONSOLIDATION_CONTEXT_MESSAGES,
            description="Maximum number of recent user messages to include in consolidation context",
        )

    def __init__(self):
        """Initialize the Memory System filter with production validation."""
        global _SHARED_SKIP_DETECTOR_CACHE

        self.valves = self.Valves()
        self._validate_system_configuration()

        self._cache_manager = UnifiedCacheManager(Constants.MAX_CACHE_ENTRIES_PER_TYPE, Constants.MAX_CONCURRENT_USER_CACHES)
        self._background_tasks: set = set()
        self._shutdown_event = asyncio.Event()

        self._embedding_function = None
        self._embedding_dimension = None
        self._skip_detector = None

        self._initialization_lock = asyncio.Lock()

        self._llm_reranking_service = LLMRerankingService(self)
        self._llm_consolidation_service = LLMConsolidationService(self)

    async def _initialize_system(self, request: Request) -> None:
        if self._embedding_function is None and hasattr(request.app.state, "EMBEDDING_FUNCTION"):
            self._embedding_function = request.app.state.EMBEDDING_FUNCTION
            logger.info("✅ Using OpenWebUI embedding function")

        if self._embedding_function and self._embedding_dimension is None:
            async with self._initialization_lock:
                if self._embedding_dimension is None:
                    await self._detect_embedding_dimension()

        if self._embedding_function and self._skip_detector is None:
            global _SHARED_SKIP_DETECTOR_CACHE, _SHARED_SKIP_DETECTOR_CACHE_LOCK
            embedding_engine = getattr(request.app.state.config, "RAG_EMBEDDING_ENGINE", "")
            embedding_model = getattr(request.app.state.config, "RAG_EMBEDDING_MODEL", "")
            cache_key = f"{embedding_engine}:{embedding_model}"

            async with _SHARED_SKIP_DETECTOR_CACHE_LOCK:
                if cache_key in _SHARED_SKIP_DETECTOR_CACHE:
                    logger.info(f"♻️ Reusing cached skip detector: {cache_key}")
                    _SHARED_SKIP_DETECTOR_CACHE.move_to_end(cache_key)
                    self._skip_detector = _SHARED_SKIP_DETECTOR_CACHE[cache_key]
                else:
                    logger.info(f"🤖 Initializing skip detector: {cache_key}")
                    embedding_fn = self._embedding_function
                    normalize_fn = self._normalize_embedding

                    async def embedding_wrapper(
                        texts: Union[str, List[str]],
                    ) -> Union[np.ndarray, List[np.ndarray]]:
                        result = await embedding_fn(texts, prefix=None, user=None)
                        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], (list, np.ndarray)):
                            return [normalize_fn(emb) for emb in result]
                        return [normalize_fn(result if isinstance(result, (list, np.ndarray)) else [result])]

                    self._skip_detector = SkipDetector(embedding_wrapper)
                    await self._skip_detector.initialize()

                    if len(_SHARED_SKIP_DETECTOR_CACHE) >= MAX_SKIP_DETECTOR_CACHE_ENTRIES:
                        _SHARED_SKIP_DETECTOR_CACHE.popitem(last=False)

                    _SHARED_SKIP_DETECTOR_CACHE[cache_key] = self._skip_detector
                    logger.info("✅ Skip detector initialized and cached")

    def _truncate_content(self, content: str, max_length: Optional[int] = None) -> str:
        """Truncate content with ellipsis if needed."""
        if max_length is None:
            max_length = Constants.CONTENT_PREVIEW_LENGTH
        return content[:max_length] + "..." if len(content) > max_length else content

    def _get_retrieval_threshold(self, is_consolidation: bool = False) -> float:
        """Calculate retrieval threshold for semantic similarity filtering."""
        if is_consolidation:
            return self.valves.semantic_retrieval_threshold * Constants.RELAXED_SEMANTIC_THRESHOLD_MULTIPLIER
        return self.valves.semantic_retrieval_threshold

    def _extract_text_from_content(self, content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text")
        if isinstance(content, dict) and "text" in content:
            return content.get("text", "")
        return ""

    def _get_recent_user_messages(self, messages: List[Dict[str, Any]], max_messages: int = 3) -> List[str]:
        """Extract recent user messages for conversation context in consolidation."""
        user_messages = []
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content", "")
            text = self._extract_text_from_content(content)
            if text:
                user_messages.append(text)
                if len(user_messages) >= max_messages:
                    break
        user_messages.reverse()
        return user_messages

    def _get_last_user_message(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract the last user message text from a list of messages."""
        recent = self._get_recent_user_messages(messages, max_messages=1)
        return recent[0] if recent else None

    def _validate_system_configuration(self) -> None:
        """Validate configuration and fail if invalid."""
        if self.valves.max_memories_returned <= 0:
            raise ValueError(f"📊 Invalid max memories returned: {self.valves.max_memories_returned}")

        if not (0.0 <= self.valves.semantic_retrieval_threshold <= 1.0):
            raise ValueError(f"🎯 Invalid semantic retrieval threshold: {self.valves.semantic_retrieval_threshold} (must be 0.0-1.0)")

        if self.valves.max_consolidation_context_messages <= 0:
            raise ValueError(f"📊 Invalid max consolidation context messages: {self.valves.max_consolidation_context_messages}")

        logger.info("✅ Configuration validated")

    async def _get_embedding_cache(self, user_id: str, key: str) -> Optional[Any]:
        """Get embedding from cache."""
        return await self._cache_manager.get(user_id, self._cache_manager.EMBEDDING_CACHE, key)

    async def _put_embedding_cache(self, user_id: str, key: str, value: Any) -> None:
        """Store embedding in cache."""
        await self._cache_manager.put(user_id, self._cache_manager.EMBEDDING_CACHE, key, value)

    def _compute_text_hash(self, text: str) -> str:
        """Compute SHA256 hash for text caching."""
        return hashlib.sha256(text.encode()).hexdigest()

    async def _detect_embedding_dimension(self) -> None:
        """Detect embedding dimension by generating a test embedding."""
        test_embedding = await self._embedding_function("dummy", prefix=None, user=None)

        if isinstance(test_embedding, list) and len(test_embedding) > 0 and isinstance(test_embedding[0], (list, np.ndarray)):
            test_embedding = test_embedding[0]

        emb_array = np.squeeze(np.array(test_embedding))
        self._embedding_dimension = emb_array.shape[0] if emb_array.ndim > 0 else 1
        logger.info(f"🎯 Detected embedding dimension: {self._embedding_dimension}")

    def _normalize_embedding(self, embedding: Union[List[float], np.ndarray]) -> np.ndarray:
        """Normalize embedding vector and ensure 1D shape."""
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float16)
        else:
            embedding = embedding.astype(np.float16)

        embedding = np.squeeze(embedding)

        if embedding.ndim != 1:
            raise ValueError(f"📐 Embedding must be 1D after squeeze, got shape {embedding.shape}")

        if self._embedding_dimension and embedding.shape[0] != self._embedding_dimension:
            raise ValueError(f"📐 Embedding dimension mismatch: expected {self._embedding_dimension}, got {embedding.shape[0]}")

        norm = np.linalg.norm(embedding)
        if norm == 0:
            logger.warning("⚠️ Zero-norm embedding detected - returning unnormalized embedding")
            return embedding
        return embedding / norm

    async def _generate_embeddings(self, texts: Union[str, List[str]], user_id: str) -> Union[np.ndarray, List[np.ndarray]]:
        """Unified embedding generation for single text or batch with optimized caching using OpenWebUI's embedding function."""
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts

        result_embeddings = []
        uncached_texts = []
        uncached_indices = []
        uncached_hashes = []

        for i, text in enumerate(text_list):
            if not text or len(text.strip()) < Constants.MIN_MESSAGE_CHARS:
                if is_single:
                    raise ValueError("📏 Text too short for embedding generation")
                result_embeddings.append(None)
                continue

            text_hash = self._compute_text_hash(text)
            cached = await self._get_embedding_cache(user_id, text_hash)

            if cached is not None:
                result_embeddings.append(cached)
            else:
                result_embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
                uncached_hashes.append(text_hash)

        if uncached_texts:
            user = await Users.get_user_by_id(user_id)
            try:
                raw_embeddings = await self._embedding_function(uncached_texts, prefix=None, user=user)
            except Exception as e:
                if isinstance(e, asyncio.CancelledError):
                    raise
                logger.error(f"❌ Embedding generation failed: {str(e)}")
                raise RuntimeError(f"Failed to generate embeddings: {str(e)}")

            if isinstance(raw_embeddings, list) and len(raw_embeddings) > 0 and isinstance(raw_embeddings[0], (list, np.ndarray)):
                new_embeddings = [self._normalize_embedding(emb) for emb in raw_embeddings]
            else:
                new_embeddings = [self._normalize_embedding(raw_embeddings)]

            for j, embedding in enumerate(new_embeddings):
                original_idx = uncached_indices[j]
                text_hash = uncached_hashes[j]
                await self._put_embedding_cache(user_id, text_hash, embedding)
                result_embeddings[original_idx] = embedding

        if is_single:
            if uncached_texts:
                logger.info("💾 User message embedding generated and cached")
            return result_embeddings[0]
        else:
            valid_count = sum(1 for emb in result_embeddings if emb is not None)
            logger.info(f"🚀 Batch embeddings: {len(text_list) - len(uncached_texts)} cached, {len(uncached_texts)} new, {valid_count}/{len(text_list)} valid")
            return result_embeddings

    async def _should_skip_memory_operations(self, user_message: str, user_id: Optional[str] = None) -> Tuple[bool, str]:
        skip_reason = await self._skip_detector.detect_skip_reason(
            user_message,
            Constants.MAX_MESSAGE_CHARS,
            memory_system=self,
            user_id=user_id,
        )
        if skip_reason:
            status_key = SkipDetector.SkipReason(skip_reason)
            return True, SkipDetector.INLET_STATUS_MESSAGES[status_key]
        return False, ""

    async def _should_skip_consolidation(self, conversation_context: List[str], user_id: Optional[str] = None) -> Tuple[bool, str]:
        """Check if consolidation should be skipped based on conversation context.

        Returns (should_skip, reason). Skips only if ALL messages in context are skippable.
        """
        logger.info(f"🔍 Evaluating {len(conversation_context)} messages for consolidation")

        for idx, message in enumerate(conversation_context, 1):
            skip_reason = await self._skip_detector.detect_skip_reason(
                message,
                Constants.MAX_MESSAGE_CHARS,
                memory_system=self,
                user_id=user_id,
            )
            if not skip_reason:  # Found at least one valuable message
                logger.info(f"✅ Found personal content in message {idx}/{len(conversation_context)}, proceeding with consolidation")
                return False, ""

        # All messages were skippable
        logger.info(f"🚫 All {len(conversation_context)} messages are non-personal, skipping consolidation")
        return True, SkipDetector.OUTLET_STATUS_MESSAGES[SkipDetector.SkipReason.SKIP_ALL_NON_PERSONAL]

    async def _process_user_message(self, body: Dict[str, Any], user_id: Optional[str] = None) -> Tuple[Optional[str], bool, str]:
        """Extract user message and determine if memory operations should be skipped."""
        user_message = self._get_last_user_message(body["messages"])
        if not user_message:
            return (
                None,
                True,
                SkipDetector.INLET_STATUS_MESSAGES[SkipDetector.SkipReason.SKIP_SIZE],
            )

        should_skip, skip_reason = await self._should_skip_memory_operations(user_message, user_id)
        return user_message, should_skip, skip_reason

    async def _get_user_memories(self, user_id: str) -> List:
        """Get user memories with timeout handling."""
        memories = await asyncio.wait_for(
            Memories.get_memories_by_user_id(user_id),
            timeout=Constants.DATABASE_OPERATION_TIMEOUT_SEC,
        )
        return [m for m in memories if m.content] if memories else []

    def _log_retrieved_memories(self, memories: List[Dict[str, Any]], context_type: str = "semantic") -> None:
        """Log retrieved memories with concise formatting showing key statistics and semantic values."""
        if not memories:
            return

        scores = [memory["relevance"] for memory in memories]
        top_score = max(scores)
        lowest_score = min(scores)
        median_score = np.median(scores)

        context_label = "📊 Consolidation candidate memories" if context_type == "consolidation" else "📊 Retrieved memories"
        max_scores_to_show = int(self.valves.max_memories_returned * Constants.EXTENDED_MAX_MEMORY_MULTIPLIER)
        scores_str = ", ".join([f"{score:.3f}" for score in scores[:max_scores_to_show]])
        suffix = "..." if len(scores) > max_scores_to_show else ""

        logger.info(f"{context_label}: {len(memories)} memories | Top: {top_score:.3f} | Median: {median_score:.3f} | Lowest: {lowest_score:.3f}")
        logger.info(f"📈 Scores: [{scores_str}{suffix}]")

    def _build_operation_details(self, created_count: int, updated_count: int, deleted_count: int) -> List[str]:
        """Build formatted operation detail strings for status messages."""
        operations = [
            (created_count, "📝 Created"),
            (updated_count, "✏️ Updated"),
            (deleted_count, "🗑️ Deleted"),
        ]
        return [f"{label} {count}" for count, label in operations if count > 0]

    def _cache_key(self, cache_type: str, user_id: str, content: Optional[str] = None) -> str:
        """Unified cache key generation for all cache types."""
        if content:
            content_hash = self._compute_text_hash(content)[: Constants.CACHE_KEY_HASH_PREFIX_LENGTH]
            return f"{cache_type}_{user_id}:{content_hash}"
        return f"{cache_type}_{user_id}"

    def _parse_timestamp(self, timestamp: Any) -> Optional[datetime]:
        """Parse various timestamp formats (epoch, ISO string, datetime) into UTC datetime."""
        if not timestamp:
            return None
        if isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        if isinstance(timestamp, str):
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                return timestamp.replace(tzinfo=timezone.utc)
            return timestamp.astimezone(timezone.utc)
        return None

    def format_current_datetime(self) -> str:
        """Return current UTC datetime in human-readable format."""
        return datetime.now(timezone.utc).strftime("%A %B %d %Y at %H:%M:%S UTC")

    def _format_memories_for_llm(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Format memories for LLM consumption with hybrid format and human-readable timestamps."""
        memory_lines = []
        for memory in memories:
            line = f"[{memory['id']}] {memory['content']}"
            record_date = memory.get("updated_at") or memory.get("created_at")
            parsed_date = self._parse_timestamp(record_date)
            if parsed_date:
                formatted_date = parsed_date.strftime("%b %d %Y")
                line += f" [noted at {formatted_date}]"
            elif record_date:
                line += f" [noted at {record_date}]"
            memory_lines.append(line)
        return memory_lines

    async def _emit_status(
        self,
        emitter: Optional[Callable],
        description: str,
        done: bool = True,
        level: int = 1,
    ) -> None:
        """Emit status messages for memory operations based on configured verbosity level."""
        if not emitter:
            return

        current_level_value = Constants.STATUS_LEVEL.get(self.valves.status_emit_level, 1)

        if current_level_value < level:
            return

        payload = {"type": "status", "data": {"description": description, "done": done}}
        result = emitter(payload)
        if asyncio.iscoroutine(result):
            await result

    async def _retrieve_relevant_memories(
        self,
        user_message: str,
        user_id: str,
        request: Request,
        user: Dict[str, Any],
        model: str,
        user_memories: Optional[List] = None,
        emitter: Optional[Callable] = None,
        cached_similarities: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Retrieve memories for injection using similarity computation with optional LLM reranking."""
        if cached_similarities is not None:
            retrieval_threshold = self._get_retrieval_threshold(is_consolidation=False)
            memories = [m for m in cached_similarities if m.get("relevance", 0) >= retrieval_threshold]
            logger.info(f"🔍 Using cached similarities: {len(memories)} candidates")
            (
                final_memories,
                reranking_info,
            ) = await self._llm_reranking_service.rerank_memories(user_message, memories, request, user, model, emitter)
            self._log_retrieved_memories(final_memories, "semantic")
            return {
                "memories": final_memories,
                "all_similarities": cached_similarities,
                "reranking_info": reranking_info,
            }

        if user_memories is None:
            user_memories = await self._get_user_memories(user_id)

        if not user_memories:
            logger.info("📭 No memories found for user")
            await self._emit_status(
                emitter,
                "📭 No Memories Found",
                done=True,
                level=Constants.STATUS_LEVEL["Intermediate"],
            )
            return {"memories": []}

        memories, all_similarities = await self._compute_similarities(user_message, user_id, user_memories)

        if memories:
            (
                final_memories,
                reranking_info,
            ) = await self._llm_reranking_service.rerank_memories(user_message, memories, request, user, model, emitter)
        else:
            logger.info("📭 No relevant memories found above similarity threshold")
            await self._emit_status(
                emitter,
                "📭 No Relevant Memories Found",
                done=True,
                level=Constants.STATUS_LEVEL["Intermediate"],
            )
            final_memories = memories
            reranking_info = {"llm_decision": False, "decision_reason": "no_candidates"}

        self._log_retrieved_memories(final_memories, "semantic")

        return {
            "memories": final_memories,
            "all_similarities": all_similarities,
            "reranking_info": reranking_info,
        }

    def _sanitize_memory_content(self, content: str) -> str:
        """Strip injection patterns and XML-breaking sequences from memory content."""
        content = content.replace("\x00", "")
        content = content.strip("<>")
        # Escape XML tag-like sequences to prevent breaking the <memory> wrapper
        content = content.replace("</memory>", "").replace("<memory>", "")
        content = content.replace("</system>", "").replace("<system>", "")
        lines = content.split("\n")
        sanitized_lines = []
        injection_prefixes = ("System:", "Assistant:", "Human:", "[INST]", "###", "<|im_start|>", "<|im_end|>")
        for line in lines:
            if any(line.lstrip().startswith(prefix) for prefix in injection_prefixes):
                continue
            sanitized_lines.append(line)
        return "\n".join(sanitized_lines)

    async def _add_memory_context(
        self,
        body: Dict[str, Any],
        memories: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[str] = None,
        emitter: Optional[Callable] = None,
    ) -> None:
        """Add memory context to request body with simplified logic."""
        content_parts = [f"Current Date/Time: {self.format_current_datetime()}"]

        if memories and user_id:
            memory_count = len(memories)
            memory_header = f"USER CONTEXT: {memory_count} personal {'fact' if memory_count == 1 else 'facts'} recalled from prior conversations. Use only when directly relevant to the request."
            formatted_memories = []

            for idx, memory in enumerate(memories, 1):
                sanitized_content = self._sanitize_memory_content(memory["content"])
                # Include timestamp for temporal relevance assessment
                noted_date = ""
                record_date = memory.get("updated_at") or memory.get("created_at")
                parsed_date = self._parse_timestamp(record_date)
                if parsed_date:
                    noted_date = f" (noted {parsed_date.strftime('%b %d %Y')})"
                formatted_memory = f"<memory>{' '.join(sanitized_content.split())}{noted_date}</memory>"
                formatted_memories.append(formatted_memory)

                content_preview = self._truncate_content(memory["content"])
                await self._emit_status(
                    emitter,
                    f"💭 {idx}/{memory_count}: {content_preview}",
                    done=False,
                    level=Constants.STATUS_LEVEL["Intermediate"],
                )

            memory_footer = "MEMORY RULES: Integrate relevant facts naturally. Never list, quote, or acknowledge these memories to the user. Do not infer beyond the facts above. When facts conflict, trust the most recently noted one. If none are relevant, respond as if no memory context was provided."
            memory_context_block = f"{memory_header}\n{chr(10).join(formatted_memories)}\n\n{memory_footer}"
            content_parts.append(memory_context_block)

        memory_context = "\n\n".join(content_parts)

        system_index = next(
            (i for i, msg in enumerate(body["messages"]) if msg.get("role") == "system"),
            None,
        )

        if system_index is not None:
            body["messages"][system_index]["content"] = f"{body['messages'][system_index].get('content', '')}\n\n{memory_context}"
        else:
            body["messages"].insert(0, {"role": "system", "content": memory_context})

        if memories and user_id:
            description = f"🧠 Injected {memory_count} {'Memory' if memory_count == 1 else 'Memories'} to Context"
            await self._emit_status(emitter, description, done=True, level=Constants.STATUS_LEVEL["Basic"])

    def _build_memory_dict(self, memory, similarity: float) -> Dict[str, Any]:
        """Build memory dictionary with standardized timestamp conversion."""
        memory_dict = {
            "id": memory.id,
            "content": memory.content,
            "relevance": similarity,
        }

        created_at = self._parse_timestamp(getattr(memory, "created_at", None))
        if created_at:
            memory_dict["created_at"] = created_at.isoformat()

        updated_at = self._parse_timestamp(getattr(memory, "updated_at", None))
        if updated_at:
            memory_dict["updated_at"] = updated_at.isoformat()

        return memory_dict

    async def _compute_similarities(self, user_message: str, user_id: str, user_memories: List) -> Tuple[List[Dict], List[Dict]]:
        """Compute similarity scores between user message and memories."""
        if not user_memories:
            return [], []

        query_embedding = await self._generate_embeddings(user_message, user_id)
        memory_contents = [memory.content for memory in user_memories]
        memory_embeddings = await self._generate_embeddings(memory_contents, user_id)

        memory_data = []
        valid_embeddings = [(i, emb) for i, emb in enumerate(memory_embeddings) if emb is not None]
        if valid_embeddings:
            indices, emb_list = zip(*valid_embeddings)
            emb_matrix = np.stack(emb_list)
            similarities = np.dot(emb_matrix, query_embedding)
            for rank, (orig_idx, sim) in enumerate(zip(indices, similarities)):
                memory_dict = self._build_memory_dict(user_memories[orig_idx], float(sim))
                memory_data.append(memory_dict)

        memory_data.sort(key=lambda x: x["relevance"], reverse=True)

        retrieval_threshold = self._get_retrieval_threshold(is_consolidation=False)
        filtered_memories = [m for m in memory_data if m["relevance"] >= retrieval_threshold]
        return filtered_memories, memory_data

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable] = None,
        __user__: Optional[Dict[str, Any]] = None,
        __request__: Optional[Request] = None,
    ) -> Dict[str, Any]:
        """Simplified inlet processing for memory retrieval and injection."""

        if not __user__ or not __request__:
            return body

        model_to_use = self.valves.memory_model or (body.get("model") if isinstance(body, dict) else None)
        if not model_to_use:
            return body

        await self._initialize_system(__request__)

        user_id = __user__.get("id")
        if not user_id:
            return body

        user_message, should_skip, skip_reason = await self._process_user_message(body, user_id)

        if not user_message or should_skip:
            if __event_emitter__ and skip_reason:
                await self._emit_status(
                    __event_emitter__,
                    skip_reason,
                    done=True,
                    level=Constants.STATUS_LEVEL["Intermediate"],
                )
            await self._add_memory_context(body, [], user_id, __event_emitter__)
            return body
        try:
            memory_cache_key = self._cache_key(self._cache_manager.MEMORY_CACHE, user_id)
            user_memories = await self._cache_manager.get(user_id, self._cache_manager.MEMORY_CACHE, memory_cache_key)
            if user_memories is None:
                user_memories = await self._get_user_memories(user_id)
                await self._cache_manager.put(
                    user_id,
                    self._cache_manager.MEMORY_CACHE,
                    memory_cache_key,
                    user_memories,
                )
            retrieval_result = await self._retrieve_relevant_memories(
                user_message,
                user_id,
                __request__,
                __user__,
                model_to_use,
                user_memories,
                __event_emitter__,
            )
            memories = retrieval_result.get("memories", [])
            all_similarities = retrieval_result.get("all_similarities", [])
            if all_similarities:
                cache_key = self._cache_key(self._cache_manager.RETRIEVAL_CACHE, user_id, user_message)
                await self._cache_manager.put(
                    user_id,
                    self._cache_manager.RETRIEVAL_CACHE,
                    cache_key,
                    all_similarities,
                )

            await self._add_memory_context(body, memories, user_id, __event_emitter__)
        except Exception as e:
            if isinstance(e, asyncio.CancelledError):
                raise
            raise RuntimeError(f"💾 Memory retrieval failed: {str(e)}")
        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable] = None,
        __user__: Optional[dict] = None,
        __request__: Optional[Request] = None,
    ) -> dict:
        """Simplified outlet processing for background memory consolidation."""

        if not __user__ or not __request__:
            return body

        model_to_use = self.valves.memory_model or (body.get("model") if isinstance(body, dict) else None)
        if not model_to_use:
            return body

        await self._initialize_system(__request__)

        user_id = __user__.get("id")
        if not user_id:
            return body

        user_message = self._get_last_user_message(body.get("messages", []))
        if not user_message:
            return body

        conversation_context = self._get_recent_user_messages(body.get("messages", []), self.valves.max_consolidation_context_messages)
        should_skip_consolidation, skip_reason = await self._should_skip_consolidation(conversation_context, user_id)

        if should_skip_consolidation:
            await self._emit_status(
                __event_emitter__,
                skip_reason,
                done=True,
                level=Constants.STATUS_LEVEL["Intermediate"],
            )
            return body

        retrieval_cache_key = self._cache_key(self._cache_manager.RETRIEVAL_CACHE, user_id, user_message)
        cached_similarities = await self._cache_manager.get(user_id, self._cache_manager.RETRIEVAL_CACHE, retrieval_cache_key)

        task = asyncio.create_task(
            self._llm_consolidation_service.run_consolidation_pipeline(
                user_message,
                user_id,
                __request__,
                __user__,
                model_to_use,
                __event_emitter__,
                cached_similarities,
                conversation_context,
            )
        )
        self._background_tasks.add(task)

        def safe_cleanup(t: asyncio.Task) -> None:
            try:
                self._background_tasks.discard(t)
                if t.cancelled():
                    return
                exception = t.exception()
                if exception:
                    logger.error(f"❌ Background consolidation failed: {str(exception)}", exc_info=exception)
                    if __event_emitter__:
                        asyncio.ensure_future(
                            self._emit_status(
                                __event_emitter__,
                                f"❌ Background consolidation failed: {str(exception)}",
                                done=True,
                                level=Constants.STATUS_LEVEL["Basic"],
                            )
                        )
            except Exception as e:
                if isinstance(e, asyncio.CancelledError):
                    raise
                logger.exception(f"❌ Failed to cleanup background task: {str(e)}")

        task.add_done_callback(safe_cleanup)
        return body

    async def shutdown(self) -> None:
        """Cleanup method to properly shutdown background tasks."""
        self._shutdown_event.set()

        if self._background_tasks:
            tasks = list(self._background_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)
            self._background_tasks.clear()

        await self._cache_manager.clear_all_caches()

    async def _refresh_user_cache(self, user_id: str, deleted_contents: Optional[List[str]] = None) -> None:
        """Refresh user cache - clear stale caches, remove deleted embeddings, and update with fresh embeddings."""
        start_time = time.time()
        try:
            retrieval_cleared = await self._cache_manager.clear_user_cache(user_id, self._cache_manager.RETRIEVAL_CACHE)

            embedding_removed = 0
            if deleted_contents:
                for content in deleted_contents:
                    if not content:
                        continue
                    content_hash = self._compute_text_hash(content)
                    if await self._cache_manager.remove(user_id, self._cache_manager.EMBEDDING_CACHE, content_hash):
                        embedding_removed += 1

            logger.info(f"🔄 Cleared cache: {retrieval_cleared} retrieval entries, {embedding_removed} embedding entries")

            user_memories = await self._get_user_memories(user_id)
            memory_cache_key = self._cache_key(self._cache_manager.MEMORY_CACHE, user_id)

            if not user_memories:
                await self._cache_manager.put(user_id, self._cache_manager.MEMORY_CACHE, memory_cache_key, [])
                logger.info("📭 No memories found for user")
                return

            await self._cache_manager.put(
                user_id,
                self._cache_manager.MEMORY_CACHE,
                memory_cache_key,
                user_memories,
            )

            duration = time.time() - start_time
            logger.info(f"🔄 Cache refreshed in {duration:.2f}s")

        except Exception as e:
            if isinstance(e, asyncio.CancelledError):
                raise
            raise RuntimeError(f"🧹 Failed to refresh cache for user {user_id} after {(time.time() - start_time):.2f}s: {str(e)}")

    async def _execute_db_operation(self, db_func: Callable, *args) -> None:
        """Execute database operation with timeout."""
        await asyncio.wait_for(
            db_func(*args),
            timeout=Constants.DATABASE_OPERATION_TIMEOUT_SEC,
        )

    async def _execute_single_operation(self, operation: Models.MemoryOperation, user: Any) -> str:
        """Execute a single memory operation."""
        try:
            if operation.operation == Models.MemoryOperationType.CREATE:
                await self._execute_db_operation(Memories.insert_new_memory, user.id, operation.content)
                return Models.MemoryOperationType.CREATE.value

            elif operation.operation == Models.MemoryOperationType.UPDATE:
                await self._execute_db_operation(
                    Memories.update_memory_by_id_and_user_id,
                    operation.id,
                    user.id,
                    operation.content,
                )
                return Models.MemoryOperationType.UPDATE.value

            elif operation.operation == Models.MemoryOperationType.DELETE:
                await self._execute_db_operation(Memories.delete_memory_by_id_and_user_id, operation.id, user.id)
                return Models.MemoryOperationType.DELETE.value
        except Exception as e:
            op_id = (operation.id or "").strip()
            content_preview = self._truncate_content(operation.content, Constants.CONTENT_PREVIEW_LENGTH) if operation.content else ""
            details = f"id={op_id}" if op_id else ""
            if content_preview:
                details = f"{details} content={content_preview}".strip()
            details = f" ({details})" if details else ""
            logger.error(f"❌ Memory DB operation failed for {operation.operation.value}{details}: {str(e)}")
            raise

    def _inline_schema_refs(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Inline $ref references in JSON schema."""
        if "$defs" not in schema:
            return schema

        defs = schema.pop("$defs")

        def _resolve(node: Any) -> Any:
            if isinstance(node, dict):
                if "$ref" in node:
                    ref = node["$ref"]
                    if ref.startswith("#/$defs/"):
                        def_name = ref.split("/")[-1]
                        if def_name in defs:
                            return _resolve(defs[def_name].copy())
                    raise ValueError(f"Unresolvable schema reference: {ref}")
                return {k: _resolve(v) for k, v in node.items()}
            elif isinstance(node, list):
                return [_resolve(item) for item in node]
            return node

        return _resolve(schema)

    async def _query_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        request: Request,
        user: Dict[str, Any],
        model: str,
        response_model: Optional[BaseModel] = None,
    ) -> Union[str, BaseModel]:
        """Query OpenWebUI's internal model system with Pydantic model parsing."""
        if not model:
            raise ValueError("🤖 No model specified for LLM operations")

        form_data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 4096,
            "stream": False,
        }

        if response_model:
            raw_schema = response_model.model_json_schema()
            schema = self._inline_schema_refs(raw_schema)
            schema["type"] = "object"
            form_data["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": schema,
                },
            }

        response = await asyncio.wait_for(
            generate_chat_completion(
                request,
                form_data,
                user=await Users.get_user_by_id(user["id"]),
            ),
            timeout=Constants.LLM_CONSOLIDATION_TIMEOUT_SEC,
        )

        if hasattr(response, "body"):
            response_data = json.loads(response.body.decode("utf-8"))
        else:
            response_data = response

        if isinstance(response_data, dict) and "choices" in response_data and isinstance(response_data["choices"], list) and len(response_data["choices"]) > 0:
            first_choice = response_data["choices"][0]
            if (
                isinstance(first_choice, dict)
                and "message" in first_choice
                and isinstance(first_choice["message"], dict)
                and "content" in first_choice["message"]
            ):
                content = first_choice["message"]["content"]
            else:
                raise ValueError("🤖 Invalid response structure: missing content in message")
        else:
            raise ValueError(f"🤖 Unexpected LLM response format: {response_data}")

        if response_model:
            try:
                parsed_data = json.loads(content)
                return response_model.model_validate(parsed_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"🔍 Invalid JSON from LLM: {str(e)}\nContent: {content}")
            except PydanticValidationError as e:
                raise ValueError(f"🔍 LLM response validation failed: {str(e)}\nContent: {content}")

        if not content or content.strip() == "":
            raise ValueError("🤖 Empty response from LLM")
        return content
