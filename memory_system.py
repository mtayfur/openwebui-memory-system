"""
title: Memory System
version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError as PydanticValidationError
from sentence_transformers import SentenceTransformer

from open_webui.utils.chat import generate_chat_completion
from open_webui.models.users import Users
from open_webui.routers.memories import Memories
from fastapi import Request

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

logger = logging.getLogger("MemorySystem")

_SHARED_MODEL_CACHE = {}

class Constants:
    """Centralized configuration constants for the memory system."""
    
    # Core System Limits
    MAX_MEMORY_CONTENT_CHARS = 500  # Character limit for LLM prompt memory content
    MAX_MEMORIES_PER_RETRIEVAL = 10  # Maximum memories returned per query
    MAX_MESSAGE_CHARS = 2500  # Maximum message length for validation
    MIN_MESSAGE_CHARS = 10  # Minimum message length for validation
    DATABASE_OPERATION_TIMEOUT_SEC = 10  # Timeout for DB operations like user lookup
    LLM_CONSOLIDATION_TIMEOUT_SEC = 60.0  # Timeout for LLM consolidation operations
    
    # Cache System
    MAX_CACHE_ENTRIES_PER_TYPE = 5000  # Maximum cache entries per cache type
    MAX_CONCURRENT_USER_CACHES = 500  # Maximum concurrent user cache instances
    CACHE_KEY_HASH_PREFIX_LENGTH = 16  # Hash prefix length for cache keys
    
    # Retrieval & Similarity
    SEMANTIC_RETRIEVAL_THRESHOLD = 0.5 # Semantic similarity threshold for retrieval
    RELAXED_SEMANTIC_THRESHOLD_MULTIPLIER = 0.9 # Multiplier for relaxed similarity threshold in secondary operations
    EXTENDED_MAX_MEMORY_MULTIPLIER = 1.5 # Multiplier for expanding memory candidates in advanced operations
    LLM_RERANKING_TRIGGER_MULTIPLIER = 0.5 # Multiplier for LLM reranking trigger threshold
    
    # Skip Detection Thresholds
    SKIP_DETECTION_SIMILARITY_THRESHOLD = 0.50  # Similarity threshold for skip category detection (tuned for zero-shot)
    SKIP_DETECTION_MARGIN = 0.05  # Minimum margin required between skip and conversational similarity to skip
    SKIP_DETECTION_CONFIDENT_MARGIN = 0.15  # Margin threshold for confident skips that trigger early exit
    
    # Safety & Operations
    MAX_DELETE_OPERATIONS_RATIO = 0.6  # Maximum delete operations ratio for safety
    MIN_OPS_FOR_DELETE_RATIO_CHECK = 6  # Minimum operations to apply ratio check
    
    # Content Display
    CONTENT_PREVIEW_LENGTH = 80  # Maximum length for content preview display    
    
    # Default Models
    DEFAULT_LLM_MODEL = "google/gemini-2.5-flash-lite"
    DEFAULT_EMBEDDING_MODEL = "Alibaba-NLP/gte-multilingual-base"

class Prompts:
    """Container for all LLM prompts used in the memory system."""

    MEMORY_CONSOLIDATION = f"""You are the Memory System Consolidator, a specialist in creating precise user memories.

## OBJECTIVE
Build precise memories of the user's personal narrative with factual, temporal statements.

## AVAILABLE OPERATIONS
- CREATE: For new, personal facts. Must be semantically and temporally enhanced.
- UPDATE: To modify existing memories, including making facts historical with a date range.
- DELETE: For explicit user requests or to resolve contradictions.
- SKIP: When no new, personal information is provided.

## PROCESSING GUIDELINES
- Personal Facts Only: Store only significant facts with lasting relevance to the user's life and identity. Exclude transient situations, questions, general knowledge, casual mentions, or momentary states.
- Maintain Temporal Accuracy:
    - Capture Dates: Record temporal information when explicitly stated or clearly derivable. Convert relative references (last month, yesterday) to specific dates.
    - Preserve History: Transform superseded facts into past-tense statements with defined time boundaries.
    - Avoid Assumptions: Do not assign current dates to ongoing states, habits, or conditions lacking explicit temporal context.
- Build Rich Entities:
    - Fuse Identifiers: Combine nouns/pronouns with specific names into a single entity.
    - Capture Relationships: Always store relationships in first-person format with complete relationship context. Never store incomplete relationships, always specify with whom.
    - Retroactive Enrichment: If a name is provided for prior entity, UPDATE only if substantially valuable.
- Ensure Memory Quality:
    - High Bar for Creation: Only CREATE memories for significant life facts, relationships, events, or core personal attributes. Skip trivial details or passing interests.
    - Contextual Completeness: Create memories that combine related information into cohesive statements. When multiple facts share connections (same topic, person, event, or timeframe), group them into a single memory rather than fragmenting. Include relevant supporting details that help understand the core fact while respecting boundaries. Only combine facts that are directly related and belong together naturally. Avoid bare statements lacking context and never merge unrelated information.
    - Mandatory Semantic Enhancement: Enhance entities with descriptive categorical nouns for better retrieval.
    - Verify Nouns/Pronouns: Link pronouns (he, she, they) and nouns to specific entities.
    - First-Person Format: Write all memories in English from the user's perspective.

## DECISION FRAMEWORK
- Selectivity: Verify the user is stating a direct, personally significant fact with lasting importance. If not, SKIP. Never create duplicate memories. Skip momentary events or casual mentions. Be conservative with CREATE and UPDATE operations.
- Strategy: Strongly prioritize enriching existing memories over creating new ones. Analyze the message holistically to identify naturally connected facts that should be captured together. When facts share connections (same person, event, situation, or causal relationship), combine them into a unified memory that preserves the complete picture. Each memory should be self-contained and meaningful.
- Execution: For new significant facts, use CREATE. For simple attribute changes, use UPDATE only if it meaningfully improves the memory. For significant changes, use UPDATE to make the old memory historical, then CREATE the new one. For contradictions, use DELETE.

## EXAMPLES (Assumes Current Date: September 15, 2025)

### Example 1
Message: "My wife Sarah loves hiking and outdoor activities. She has an active lifestyle and enjoys rock climbing. I started this new hobby last month and it's been great."
Memories: []
Return: {{"ops": [{{"operation": "CREATE", "id": "", "content": "My wife Sarah has an active lifestyle and enjoys hiking, outdoor activities, and rock climbing"}}, {{"operation": "CREATE", "id": "", "content": "I started rock climbing in August 2025 as a new hobby and have been enjoying it"}}]}}
Explanation: Multiple facts about the same person (Sarah's active lifestyle, love for hiking, outdoor activities, and rock climbing) are combined into a single cohesive memory. The user's separate rock climbing hobby is kept as a distinct memory since it's about a different person.

### Example 2
Message: "My daughter Emma just turned 12. We adopted a dog named Max for her 11th birthday. What should I give her for her 12th birthday?"
Memories: [id:mem-002] My daughter Emma is 10 years old [noted at March 20 2024] [id:mem-101] I have a golden retriever [noted at September 20 2024]
Return: {{"ops": [{{"operation": "UPDATE", "id": "mem-002", "content": "My daughter Emma turned 12 years old in September 2025"}}, {{"operation": "UPDATE", "id": "mem-101", "content": "I have a golden retriever named Max that was adopted in September 2024 as a birthday gift for my daughter Emma when she turned 11"}}]}}
Explanation: Dog memory enriched with related context (Emma, birthday gift, age 11) and temporal anchoring (September 2024) - all semantically connected to the same event and relationship.

### Example 3
Message: "Can you recommend some good tapas restaurants in Barcelona? I moved here from Madrid last month."
Memories: [id:mem-005] I live in Madrid Spain [noted at June 12 2025]
Return: {{"ops": [{{"operation": "UPDATE", "id": "mem-005", "content": "I lived in Madrid Spain until August 2025"}}, {{"operation": "CREATE", "id": "", "content": "I moved to Barcelona Spain in August 2025"}}]}}
Explanation: Relocation is a significant life event with lasting impact. "Exploring the city" and "adjusting" are transient states and excluded.

### Example 4
Message: "My wife Sofia and I just got married in August. What are some good honeymoon destinations?"
Memories: [id:mem-008] I am single [noted at January 5 2025]
Return: {{"ops": [{{"operation": "DELETE", "id": "mem-008", "content": ""}}, {{"operation": "CREATE", "id": "", "content": "I married Sofia in August 2025 and she is now my wife"}}]}}
Explanation: Marriage is an enduring life event. Wife's name and marriage date are lasting facts combined naturally. "Planning honeymoon" is a transient activity and excluded.

### Example 5
Message: "¡Hola! Me mudé de Madrid a Barcelona el mes pasado y me casé con mi novia Sofía en agosto. ¿Me puedes recomendar un buen restaurante para celebrar?"
Memories: [id:mem-005] I live in Madrid Spain [noted at June 12 2025] [id:mem-006] I am dating Sofia [noted at February 10 2025] [id:mem-008] I am single [noted at January 5 2025]
Return: {{"ops": [{{"operation": "UPDATE", "id": "mem-005", "content": "I lived in Madrid Spain until August 2025"}}, {{"operation": "DELETE", "id": "mem-008", "content": ""}}, {{"operation": "UPDATE", "id": "mem-006", "content": "I moved to Barcelona Spain and married my girlfriend Sofia in August 2025, who is now my wife"}}]}}
Explanation: The user's move and marriage are significant, related life events that occurred in the same month. They are consolidated into a single, cohesive memory that enriches the existing relationship context.

### Example 6
Message: "I'm feeling stressed about work this week and looking for some relaxation tips. I have a big presentation coming up on Friday."
Memories: []
Return: {{"ops": []}}
Explanation: Temporary stress, seeking tips, and upcoming presentation are all transient situations without lasting personal significance. Nothing to store.
"""

    MEMORY_RERANKING = f"""You are the Memory Relevance Analyzer.

## OBJECTIVE
Select relevant memories to personalize the response, prioritizing direct connections and supporting context.

## RELEVANCE CATEGORIES
- Direct: Memories explicitly about the query topic, people, or domain.
- Contextual: Personal info that affects response recommendations or understanding.
- Background: Situational context that provides useful personalization.

## SELECTION FRAMEWORK
- Prioritize Current Info: Give current facts higher relevance than historical ones unless the query is about the past or historical context directly informs the current situation.
- Hierarchy: Prioritize Direct → Contextual → Background.
- Ordering: Order IDs by relevance, most relevant first.
- Standard: Prioritize topic matches, then context that enhances the response.
- Maximum Limit: Return up to {Constants.MAX_MEMORIES_PER_RETRIEVAL} memory IDs.

## EXAMPLES (Assumes Current Date: September 15, 2025)

### Example 1
Message: "I'm struggling with imposter syndrome at my new job. Any advice?"
Memories: [id:mem-001] I work as a senior software engineer at Tesla [noted at September 10 2025] [id:mem-002] I started my current job 3 months ago [noted at June 15 2025] [id:mem-003] I used to work in marketing [noted at March 5 2025] [id:mem-004] I graduated with a computer science degree [noted at May 15 2020]
Return: {{"ids": ["mem-001", "mem-002", "mem-003", "mem-004"]}}
Explanation: Career transition history (marketing → software engineering) directly informs current imposter syndrome at new job, making historical context relevant.

### Example 2
Message: "Necesito ideas para una cena saludable y con muchas verduras esta noche."
Memories: [id:mem-030] I am trying a vegetarian diet [noted at September 20 2025] [id:mem-031] My favorite cuisine is Italian [noted at August 15 2025] [id:mem-032] I dislike spicy food [noted at August 5 2025]
Return: {{"ids": ["mem-030", "mem-031", "mem-032"]}}
Explanation: Vegetarian diet is directly relevant to healthy vegetable-focused dinner. Italian cuisine and spice preference provide contextual personalization for recipe recommendations.

### Example 3
Message: "What are some good anniversary gift ideas for my wife, Sarah?"
Memories: [id:mem-101] My wife is named Sarah. [id:mem-102] My wife Sarah loves hiking and mystery novels. [id:mem-103] My wedding anniversary with Sarah is in October. [id:mem-104] I am on a tight budget this month. [id:mem-105] I live in Denver. [id:mem-106] I have a golden retriever named Max.
Return: {{"ids": ["mem-102", "mem-103", "mem-101", "mem-104"]}}
Explanation: Wife's interests (hiking, mystery novels) are direct matches for gift suggestions. Anniversary timing and budget constraints are contextual factors. Location and pet are background details not relevant to gift selection.

### Example 4
Message: "I've been reading about quantum computing and I'm confused. Can you break down how quantum bits work differently from regular computer bits?"
Memories: [id:mem-026] I work as a senior software engineer at Tesla [noted at September 15 2025] [id:mem-027] My wife is named Sarah [noted at August 5 2025]
Return: {{"ids": []}}
Explanation: Query seeks general technical explanation without personal context. Job and family information don't affect how quantum computing concepts should be explained.
"""

class Models:
    """Container for all Pydantic models used in the memory system."""

    class MemoryOperationType(Enum):
        CREATE = "CREATE"
        UPDATE = "UPDATE"
        DELETE = "DELETE"

    class OperationResult(Enum):
        SKIPPED_EMPTY_CONTENT = "SKIPPED_EMPTY_CONTENT"
        SKIPPED_EMPTY_ID = "SKIPPED_EMPTY_ID"
        UNSUPPORTED = "UNSUPPORTED"
        FAILED = "FAILED"

    class StrictModel(BaseModel):
        """Base model with strict JSON schema for LLM structured output."""

        model_config = ConfigDict(extra="forbid")

    class MemoryOperation(StrictModel):
        """Pydantic model for memory operations with validation."""

        operation: 'Models.MemoryOperationType' = Field(description="Type of memory operation to perform")
        content: str = Field(description="Memory content (required for CREATE/UPDATE, empty for DELETE)")
        id: str = Field(description="Memory ID (empty for CREATE, required for UPDATE/DELETE)")

        def validate_operation(self, existing_memory_ids: Optional[set] = None) -> bool:
            """Validate the memory operation against existing memory IDs."""
            if existing_memory_ids is None:
                existing_memory_ids = set()

            if self.operation == Models.MemoryOperationType.CREATE:
                return True
            elif self.operation in [Models.MemoryOperationType.UPDATE, Models.MemoryOperationType.DELETE]:
                return self.id in existing_memory_ids
            return False

    class ConsolidationResponse(BaseModel):
        """Pydantic model for memory consolidation LLM response - object containing array of memory operations."""

        ops: List['Models.MemoryOperation'] = Field(default_factory=list, description="List of memory operations to execute")

    class MemoryRerankingResponse(BaseModel):
        """Pydantic model for memory reranking LLM response - object containing array of memory IDs."""

        ids: List[str] = Field(default_factory=list, description="List of memory IDs selected as most relevant for the user query")


class UnifiedCacheManager:
    """Unified cache manager handling all cache types with user isolation and LRU eviction."""

    def __init__(self, max_cache_size_per_type: int, max_users: int):
        self.max_cache_size_per_type = max_cache_size_per_type
        self.max_users = max_users
        self.caches: OrderedDict[str, Dict[str, OrderedDict[str, Any]]] = OrderedDict()
        self._lock = asyncio.Lock()

        self.EMBEDDING_CACHE = "embedding"
        self.RETRIEVAL_CACHE = "retrieval"
        self.MEMORY_CACHE = "memory"

    async def get(self, user_id: str, cache_type: str, key: str) -> Optional[Any]:
        """Get value from cache with LRU updates."""
        async with self._lock:
            if user_id not in self.caches:
                return None

            user_cache = self.caches[user_id]
            if cache_type not in user_cache:
                return None

            type_cache = user_cache[cache_type]
            if key in type_cache:
                type_cache.move_to_end(key)
                self.caches.move_to_end(user_id)
                return type_cache[key]
            return None

    async def put(self, user_id: str, cache_type: str, key: str, value: Any) -> None:
        """Store value in cache with size limits and LRU eviction."""
        async with self._lock:
            if user_id not in self.caches:
                if len(self.caches) >= self.max_users:
                    evicted_user, _ = self.caches.popitem(last=False)
                self.caches[user_id] = {}

            user_cache = self.caches[user_id]

            if cache_type not in user_cache:
                user_cache[cache_type] = OrderedDict()

            type_cache = user_cache[cache_type]

            if key not in type_cache and len(type_cache) >= self.max_cache_size_per_type:
                evicted_key, _ = type_cache.popitem(last=False)

            if key in type_cache:
                type_cache[key] = value
                type_cache.move_to_end(key)
            else:
                type_cache[key] = value

            self.caches.move_to_end(user_id)

    async def clear_user_cache(self, user_id: str, cache_type: Optional[str] = None) -> int:
        """Clear specific cache type for user, or all caches for user if cache_type is None."""
        async with self._lock:
            if user_id not in self.caches:
                return 0

            user_cache = self.caches[user_id]

            if cache_type is None:
                total_cleared = sum(len(type_cache) for type_cache in user_cache.values())
                del self.caches[user_id]
                return total_cleared
            else:
                if cache_type in user_cache:
                    cleared_count = len(user_cache[cache_type])
                    del user_cache[cache_type]

                    if not user_cache:
                        del self.caches[user_id]

                    return cleared_count
                return 0

    async def clear_all_caches(self) -> None:
        """Clear all caches for all users."""
        async with self._lock:
            self.caches.clear()

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        async with self._lock:
            total_users = len(self.caches)
            total_items = 0
            cache_type_counts = {}

            for user_id, user_cache in self.caches.items():
                for cache_type, type_cache in user_cache.items():
                    cache_type_counts[cache_type] = cache_type_counts.get(cache_type, 0) + len(type_cache)
                    total_items += len(type_cache)

            return {
                "total_users": total_users,
                "total_items": total_items,
                "cache_type_counts": cache_type_counts,
                "max_users": self.max_users,
                "max_cache_size_per_type": self.max_cache_size_per_type,
            }


class SkipDetector:
    """Semantic-based content classifier using zero-shot classification with category descriptions."""

    TECHNICAL_CATEGORY_DESCRIPTIONS = [
        "programming language syntax, implementation, algorithm logic, function, method, class, object-oriented, variable scope, data type, control flow, import, module, package, library, framework, recursion, iteration, memoization, paradigm",
        "software design patterns, creational: singleton, factory, builder, prototype; structural: adapter, decorator, facade, proxy; behavioral: observer, strategy, command, mediator, chain of responsibility; interface, polymorphism, composition",
        "error handling, exception, traceback, stack trace, TypeError, NullPointerException, IndexError, segmentation fault, core dump, stack overflow, runtime vs compile-time error, assertion failed, syntax error, undefined reference, memory leak",
        "HTTP status codes: 404 Not Found, 500 Internal Server Error, 403 Forbidden, 401 Unauthorized, 200 OK, 201 Created. API response, 502 Bad Gateway, 503 Service Unavailable, 400 Bad Request, 429 Too Many Requests, timeout, CORS",
        "terminal command line shell prompt, bash, zsh, powershell, cmd. Filesystem navigation: cd, ls, pwd. File management: mkdir, rm, cp, mv, chmod, chown. Text processing: grep, sed, awk, cat, echo. User permissions: sudo, root",
        "developer CLI tools, package manager, install, update. Network requests with curl, wget. Secure shell access with SSH. Version control with git: clone, commit, push, pull. Containerization with docker: run, build, compose; npm, pip",
        "data interchange formats, serialization, deserialization, parsing. JSON object, array, key-value pair. XML tags, attributes. YAML indentation, TOML, CSV, .ini properties. Config files, env variables, dictionary, map, protocol buffers",
        "WebSocket real-time bidirectional communication, server-client connection on a port, binary message protocol, handshake, upgrading from HTTP, socket programming, TCP, UDP, listening, binding, accepting, streaming, pub-sub, broadcast channel",
        "API design, endpoint, REST, GraphQL, SOAP, RPC. HTTP methods: GET, POST, PUT, DELETE, PATCH. Request-response cycle, payload, authentication, authorization token, bearer, JWT, OAuth, API key, query parameters, path variables, body",
        "file system paths, directory structure, config files, log files, binary executables, absolute vs relative paths, operating system, mount point, home, /tmp, /var, /usr, shared library, symbolic link, inode, file permissions, owner, group",
        "algorithm analysis, O(log n) time complexity, space complexity, data structures, hash table, array, linked list, queue, stack, heap, priority queue, graph, adjacency matrix, depth-first search (DFS), breadth-first search (BFS)",
        "sorting algorithms performance and implementation, including merge sort, quicksort, insertion sort, selection sort, bubble sort, heap sort. Understanding stable vs unstable sorts, in-place operations, and comparison-based sorting concepts",
        "markdown syntax for text formatting, horizontal rule, separator using dashes or asterisks, headings, fenced code blocks with triple backticks, inline code, emphasis with bold and italic, strikethrough, blockquote, nested lists, task list",
        "code formatting and style, indentation with whitespace, tabs vs spaces, nested function bodies, class methods, structured code, syntax highlighting for languages like Python, JavaScript, Java, C++, Go, Rust, TypeScript, Prettier, ESLint",
        "container orchestration, cluster management, service scaling, replication, load balancing, namespace, pod, deployment, infrastructure, Kubernetes (K8s), Docker Swarm, Mesos, container runtime (CRI-O, containerd), image registry, Dockerfile",
        "database query, SQL statement, table, column, row, index, primary key, foreign key, relationship, join, filter, select, insert, update, delete, relational vs NoSQL, MongoDB, PostgreSQL, MySQL, Redis, schema, normalization, transaction",
        "application logging, log output, stack trace levels like INFO, WARN, ERROR, DEBUG, FATAL. Log message components: timestamp, level, module, line number. Diagnostic telemetry, monitoring, and observability for system health and debugging",
        "regular expression (regex) patterns, matching groups, capturing, backslash escapes, metacharacters, wildcards, quantifiers, character classes, lookaheads, lookbehinds, alternation, anchors, word boundaries, multiline flag, global search",
        "software testing, unit test, assertion, mock, stub, fixture, test suite, test case, verification, automated QA, validation framework, JUnit, pytest, Jest. Integration, end-to-end (E2E), functional, regression, acceptance testing",
        "cloud computing platforms, infrastructure as a service (IaaS), platform as a service (PaaS), AWS, Azure, GCP, compute instance, region, availability zone, scaling, elasticity, distributed system, virtual machine, serverless, Lambda",
    ]

    INSTRUCTION_CATEGORY_DESCRIPTIONS = [
        "format the output as a structured data type. Return the answer as JSON with specific keys and values, or as YAML. Organize the information into a CSV file or a database-style table with specific columns and rows. Present a list of objects.",
        "style the presentation of the text. Use markdown formatting like bullet points, a numbered list, or a task list. Organize the content into a grid or tabular layout with proper alignment. Create a hierarchical structure with nested elements.",
        "adjust the length of the response. Make the answer shorter, more concise, brief, or condensed. Summarize the key points. Trim down the text to reduce the overall word count or meet a specific character limit. Be less verbose and more direct.",
        "change the depth of the explanation. Make the response more detailed, comprehensive, and elaborate. Expand on the previous points and go into more depth. Provide a thorough, in-depth analysis. Explain the topic with more complexity and nuance.",
        "rewrite the previous response. Rephrase, paraphrase, or reformulate the answer using different wording. Restate the information in another way to offer an alternative perspective. Express the same meaning but with a new structure and vocabulary.",
        "alter the tone of the response. Change the writing style to be more formal, academic, or professional. Alternatively, make it more casual, friendly, and conversational. Adapt the register and voice to suit a specific audience or context.",
        "explain the concept in simpler terms. Break down the topic step-by-step for a beginner. Clarify a confusing point. Explain it like I'm five years old (ELI5). Use an analogy or provide a concrete example to help me understand the idea.",
        "continue the response. Keep going with the explanation or list. Provide more information and finish your thought. Complete the rest of the content or story. Proceed with the next steps. Do not stop until you have reached the conclusion.",
        "act as a specific persona or role. Respond as if you were a pirate, a scientist, a travel guide, or a Shakespearean actor. Adopt the character's voice, style, and knowledge base in your answer. Maintain the persona throughout the response.",
        "compare and contrast two or more topics. Explain the similarities and differences between A and B. Provide a detailed analysis of what they have in common and how they diverge. Create a table to highlight the key distinctions and resemblances.",
    ]

    PURE_MATH_CALCULATION_CATEGORY_DESCRIPTIONS = [
        "perform a pure arithmetic calculation with explicit numbers. Solve, multiply, add, subtract, and divide. Compute a numeric expression following the order of operations (PEMDAS/BODMAS). What is 23 plus 456 minus 78 times 9 divided by 3?",
        "evaluate a mathematical expression containing numbers and operators, such as 2 plus 3 times 4 divided by 5. Solve this numerical problem and compute the final result. Simplify the arithmetic and show the final answer. Calculate 123 * 456.",
        "convert units between measurement systems with numeric values. Convert 100 kilometers to miles, 72 fahrenheit to celsius, or 5 feet 9 inches to centimeters. Change between metric and imperial for distance, weight, volume, or temperature.",
        "calculate a percentage of a number. What is 25 percent of 800? Determine the price after a 30% discount. Compute a 15% tip on a $65.40 bill. Find the value corresponding to a specific proportion or calculate sales tax or interest.",
        "solve an algebraic equation for a variable like x. For the equation 2x + 5 = 15, find the value of x. Use the quadratic formula for numeric values. Solve simultaneous linear equations to find the value of the unknown variables. Isolate x.",
        "perform a geometry calculation with numeric measurements. Find the area of a circle with a radius of 5, or the volume of a cube with a side of 10. Calculate the circumference, perimeter, or diameter. What is the square root of 144?",
        "calculate compound interest on an investment or savings. With a principal of $5000 at an annual rate of 4% for 10 years, what is the future value? Compute a monthly mortgage payment for a $300,000 loan. Financial calculation, ROI, APR.",
        "compute descriptive statistics for a dataset of numbers like 12, 15, 18, 20, 22. Calculate the mean, median, mode, average, and standard deviation. Find the variance, range, quartiles, and percentiles for a given sample distribution.",
        "calculate health and fitness metrics using a numeric formula. Compute the Body Mass Index (BMI) given a weight in pounds or kilograms and height in feet, inches, or meters. Find my basal metabolic rate (BMR) or target heart rate.",
        "calculate the time difference between two dates. How many days, hours, or minutes are between two points in time? Find the duration or elapsed time. Act as an age calculator for a birthday or find the time until a future anniversary.",
    ]

    EXPLICIT_TRANSLATION_CATEGORY_DESCRIPTIONS = [
        "translate the explicitly quoted text 'Hello, how are you?' to a foreign language like Spanish, French, or German. This is a translation instruction that includes the word 'translate' and the source text in quotes for direct conversion.",
        "how do you say a specific word or phrase in another language? For example, how do you say 'thank you', 'computer', or 'goodbye' in Japanese, Chinese, or Korean? This is a request for a direct translation of a common expression or term.",
        "convert a block of text or a paragraph from a source language to a target language. Translate the following content to Italian, Arabic, Portuguese, or Russian. This is a language conversion request for a larger piece of text provided.",
        "provide the translation for the sentence 'Where is the train station?' into a specific foreign language like Turkish, Hindi, or Polish. This is a translation request for a complete sentence, often enclosed in quotes or brackets for clarity.",
        "what is the translation of the source text 'The quick brown fox jumps over the lazy dog' into a target language? Give me the resulting translated output in German, French, or Dutch. This is a query for the translated equivalent of a text.",
        "translate the following passage to Spanish. This is an instruction to convert the provided text content into a specified foreign language. The request uses a direct command format, indicating a clear source and a clear target language.",
        "what is the foreign language word for 'house', 'beautiful', or 'water'? Provide the translation for these common vocabulary words in Italian, Swedish, or another language. This is a request for single-word vocabulary translation.",
        "how do I say 'I am learning to code' in German? Convert this specific English phrase into its equivalent in another language. This is a request for a practical, conversational phrase translation for personal or professional use.",
        "translate this informal or slang expression to its colloquial equivalent in Spanish. How would you say 'What's up?' in Japanese in a casual context? This request focuses on capturing the correct tone and nuance of informal language.",
        "provide the formal and professional translation for 'Please find the attached document for your review' in French. Translate this business email phrase to German, ensuring the terminology and register are appropriate for a corporate context.",
    ]

    GRAMMAR_PROOFREADING_CATEGORY_DESCRIPTIONS = [
        "proofread the following text for errors. Here is my draft, please check it for typos and mistakes: 'Teh quick brown fox jumpped'. Review, revise, and correct any misspellings or grammatical issues you find in the provided passage.",
        "correct the grammar in this specific sentence: 'She don't like it' or 'Their going too the store'. Fix issues with subject-verb agreement, incorrect verb tense, pronoun reference errors, or misplaced modifiers in the provided text.",
        "check the spelling and punctuation in this passage. Please review the following text and fix any errors: 'its a beautiful day, isnt it'. Correct missing commas, periods, apostrophes, quotation marks, colons, or capitalization.",
        "review this sentence and tell me if it is grammatically correct. Is the sentence 'There going to they're house' proper? Validate the grammar, check word usage (like their/there/they're), and verify that the sentence is well-formed.",
        "proofread my email before I send it. Here is the draft. Please check for clarity, flow, coherence, and readability. Improve my writing, make it better, and polish the text to ensure it sounds professional and is free of errors.",
        "fix the punctuation in this run-on sentence or comma splice. Correct sentence fragments and ensure proper use of capitalization. Address errors with apostrophes, quotation marks, periods, semicolons, dashes, and other punctuation marks.",
        "suggest a better word choice or alternative phrasing. Can you help me improve my vocabulary and diction in this sentence? Replace words with more precise or impactful synonyms. Refine the expression for better clarity, tone, or style.",
        "rewrite this sentence from passive voice to active voice. Help me make my writing more direct and concise by eliminating passive constructions. Restructure the sentence to be more engaging and clear. Identify and fix faulty parallelism.",
        "improve the clarity and flow of this paragraph. Make the writing smoother and more readable. Restructure the sentences for better coherence and logical progression. Ensure the ideas connect seamlessly and eliminate any awkward phrasing.",
        "check my essay for conciseness and remove any redundancy. Help me edit this text to be more direct and to the point. Identify and eliminate wordiness, filler words, and repetitive phrases to strengthen the overall quality of the writing.",
    ]

    CONVERSATIONAL_CATEGORY_DESCRIPTIONS = [
        "discussing my family members, like my spouse, children, parents, or siblings. Mentioning relatives by name or role, such as my husband, wife, son, daughter, mother, or father. Sharing stories or asking questions about my family.",
        "expressing lasting personal feelings, core values, beliefs, or principles. My worldview, deeply held opinions, philosophy, or moral standards. Things I love, hate, or feel strongly about in life, such as my passion for animal welfare.",
        "describing my established personal hobbies, regular activities, or consistent interests. My passions and what I do in my leisure time, such as creative outlets like painting, sports like hiking, or other recreational pursuits I enjoy.",
        "sharing information about my career or current job. My position, workplace, company name, or professional role. My responsibilities at work, my occupation, or the industry I work in. My employment situation, job title, and employer.",
        "talking about my major life plans, long-term aspirations, or personal goals. My dreams for the future, important intentions, and what I want to achieve. Milestones, ambitions, or a bucket list. My personal vision or mission in life.",
        "reflecting on a meaningful personal story, memory, or significant past life experience. A transformative event or milestone that shaped me. A defining moment, a lesson learned from my childhood, or a memory from growing up that I cherish.",
        "sharing my personal background, like my hometown, childhood upbringing, or education. My cultural heritage, ethnicity, or where I grew up. Information about the university I graduated from or formative life experiences that define my identity.",
        "asking for personal advice about a specific life situation, relationship, family decision, or career choice. Seeking guidance on a personal challenge, problem, or dilemma I'm facing. Needing help or counsel on a difficult issue or conflict.",
        "requesting personalized recommendations based on my stated context, preferences, or needs. For example, suggesting a movie based on genres I like, or a restaurant that fits my dietary restrictions, budget, and location requirements.",
        "talking about my personal learning journey or educational pursuits. A course I'm taking, a certification I'm working on, or a degree program I'm enrolled in. My efforts in personal development, skill acquisition, or knowledge building.",
        "discussing my child, spouse, or other family member's interests or needs. Helping my son with a school project, finding a hobby for my daughter, or supporting my partner's career goals. Questions related to supporting my loved ones.",
        "describing my personal challenges with a work task, learning a new skill, or a technology problem. Feeling confused, stressed, or overwhelmed. Dealing with imposter syndrome, self-doubt, or needing assistance with a difficult project.",
        "planning a personal event like a party, celebration, or family gathering. Organizing my daughter's birthday, my son's graduation, or a wedding anniversary. Discussing arrangements for a special occasion or festive milestone commemoration.",
        "mentioning my pet, such as my dog, cat, or another animal companion. I adopted a puppy, or I have a cat named Luna. Discussing my pet's breed, age, behavior, or my general feelings about animals, pet care, and pet ownership.",
        "discussing moving or relocating to a new city, state, or country. I just moved into a new apartment or house. The personal reasons for my move, like a job or family. The process of settling into a new home, neighborhood, or location.",
        "stating my long-term dietary preference or restriction, such as being vegetarian, vegan, pescatarian, gluten-free, or having a food allergy. My eating habits and favorite cuisines, based on health, ethical, or personal reasons.",
        "talking about my religious or cultural practices. I celebrate Christmas, observe Ramadan, or follow Buddhist traditions. My faith, beliefs, spirituality, or cultural background. Religious identity, worship, prayers, rituals, or holidays.",
        "describing my living situation. I live with roommates, alone, with my parents, or with a partner. I bought or rented a house or apartment. My home environment, housing arrangements, and household composition in my current residence.",
        "talking about my personal finances, such as saving for a down payment on a house, managing a tight budget, or planning for retirement. My investment goals, strategies for handling debt, or my general approach to financial security.",
        "working on a personal creative project. I am writing a novel, composing music, painting a picture, or developing a side project. A meaningful creative pursuit or hobby that involves a personal, emotional investment and artistic expression.",
        "describing my fitness routine or exercise habits. I go to the gym, run, do yoga, or swim regularly. My consistent activities for health and wellness, my workout regimen, or my training schedule and fitness goals for an active lifestyle.",
        "sharing my personal values and what I care about deeply. I believe strongly in environmental sustainability, social justice, or equality. Causes I support, my principles, ethics, morals, and convictions that shape my worldview and priorities.",
        "discussing a personal achievement or milestone. I got promoted, received an award, won a competition, or completed a marathon. A significant accomplishment I am proud of, a goal I reached, or a success that marked a personal triumph.",
        "referencing my social preferences. I am an introvert, an extrovert, or an ambivert. I prefer small groups over large crowds. My personality trait regarding socializing, my interaction style, and where I get my energy in social settings.",
    ]

    class SkipReason(Enum):
        SKIP_SIZE = "SKIP_SIZE"
        SKIP_TECHNICAL = "SKIP_TECHNICAL"
        SKIP_INSTRUCTION = "SKIP_INSTRUCTION"
        SKIP_PURE_MATH = "SKIP_PURE_MATH"
        SKIP_TRANSLATION = "SKIP_TRANSLATION"
        SKIP_GRAMMAR_PROOFREAD = "SKIP_GRAMMAR_PROOFREAD"

    STATUS_MESSAGES = {
        SkipReason.SKIP_SIZE: "📏 Message Length Out of Limits, skipping memory operations",
        SkipReason.SKIP_TECHNICAL: "💻 Technical Content Detected, skipping memory operations",
        SkipReason.SKIP_INSTRUCTION: "💬 Instruction Detected, skipping memory operations",
        SkipReason.SKIP_PURE_MATH: "🔢 Mathematical Calculation Detected, skipping memory operations",
        SkipReason.SKIP_TRANSLATION: "🌐 Translation Request Detected, skipping memory operations",
        SkipReason.SKIP_GRAMMAR_PROOFREAD: "📝 Grammar/Proofreading Request Detected, skipping memory operations",
    }

    def __init__(self, embedding_model: SentenceTransformer):
        """Initialize the skip detector with an embedding model and compute reference embeddings."""
        self.embedding_model = embedding_model
        self._reference_embeddings = None
        self._initialize_reference_embeddings()
    
    def _initialize_reference_embeddings(self) -> None:
        """Compute and cache embeddings for category descriptions."""
        try:
            technical_embeddings = self.embedding_model.encode(
                self.TECHNICAL_CATEGORY_DESCRIPTIONS, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            instruction_embeddings = self.embedding_model.encode(
                self.INSTRUCTION_CATEGORY_DESCRIPTIONS,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            pure_math_embeddings = self.embedding_model.encode(
                self.PURE_MATH_CALCULATION_CATEGORY_DESCRIPTIONS,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            translation_embeddings = self.embedding_model.encode(
                self.EXPLICIT_TRANSLATION_CATEGORY_DESCRIPTIONS,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            grammar_embeddings = self.embedding_model.encode(
                self.GRAMMAR_PROOFREADING_CATEGORY_DESCRIPTIONS,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            conversational_embeddings = self.embedding_model.encode(
                self.CONVERSATIONAL_CATEGORY_DESCRIPTIONS,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            self._reference_embeddings = {
                'technical': technical_embeddings,
                'instruction': instruction_embeddings,
                'pure_math': pure_math_embeddings,
                'translation': translation_embeddings,
                'grammar': grammar_embeddings,
                'conversational': conversational_embeddings,
            }
            
            total_skip_categories = (
                len(self.TECHNICAL_CATEGORY_DESCRIPTIONS) + 
                len(self.INSTRUCTION_CATEGORY_DESCRIPTIONS) +
                len(self.PURE_MATH_CALCULATION_CATEGORY_DESCRIPTIONS) +
                len(self.EXPLICIT_TRANSLATION_CATEGORY_DESCRIPTIONS) +
                len(self.GRAMMAR_PROOFREADING_CATEGORY_DESCRIPTIONS)
            )
            
            logger.info(f"SkipDetector initialized with {total_skip_categories} skip categories and {len(self.CONVERSATIONAL_CATEGORY_DESCRIPTIONS)} personal categories")
        except Exception as e:
            logger.error(f"Failed to initialize SkipDetector reference embeddings: {e}")
            self._reference_embeddings = None

    def validate_message_size(self, message: str, max_message_chars: int) -> Optional[str]:
        """Validate message size constraints."""
        if not message or not message.strip():
            return SkipDetector.SkipReason.SKIP_SIZE.value
        trimmed = message.strip()
        if len(trimmed) < Constants.MIN_MESSAGE_CHARS or len(trimmed) > max_message_chars:
            return SkipDetector.SkipReason.SKIP_SIZE.value
        return None

    def _fast_path_skip_detection(self, message: str) -> Optional[str]:
        """Language-agnostic structural pattern detection with high confidence and low false positive rate."""
        msg_len = len(message)
        
        # Pattern 1: Multiple URLs (5+ full URLs indicates link lists or technical references)
        url_pattern_count = message.count('http://') + message.count('https://')
        if url_pattern_count >= 5:
            return self.SkipReason.SKIP_TECHNICAL.value
        
        # Pattern 2: Long unbroken alphanumeric strings (tokens, hashes, base64)
        words = message.split()
        for word in words:
            cleaned = word.strip('.,;:!?()[]{}"\'"')
            if len(cleaned) > 80 and cleaned.replace('-', '').replace('_', '').isalnum():
                return self.SkipReason.SKIP_TECHNICAL.value
        
        # Pattern 3: Markdown/text separators (repeated ---, ===, ___, ***)
        separator_patterns = ['---', '===', '___', '***']
        for pattern in separator_patterns:
            if message.count(pattern) >= 2:
                return self.SkipReason.SKIP_TECHNICAL.value
        
        # Pattern 4: Command-line patterns with context-aware detection
        lines_stripped = [line.strip() for line in message.split('\n') if line.strip()]
        if lines_stripped:
            actual_command_lines = 0
            for line in lines_stripped:
                if line.startswith('$ ') and len(line) > 2:
                    parts = line[2:].split()
                    if parts and parts[0].isalnum():
                        actual_command_lines += 1
                # Check for lines with embedded $ commands (e.g., "Run: $ command")
                elif '$ ' in line:
                    dollar_index = line.find('$ ')
                    if dollar_index > 0 and line[dollar_index-1] in (' ', ':', '\t'):
                        parts = line[dollar_index+2:].split()
                        if parts and len(parts[0]) > 0 and (parts[0].isalnum() or parts[0] in ['curl', 'wget', 'git', 'npm', 'pip', 'docker']):
                            actual_command_lines += 1
                elif line.startswith('# ') and len(line) > 2:
                    rest = line[2:].strip()
                    if rest and not rest[0].isupper() and ' ' in rest:
                        actual_command_lines += 1
                elif line.startswith('> ') and len(line) > 2:
                    pass
            
            # Lowered threshold: even 1 command line with URL/pipe is technical
            if actual_command_lines >= 1 and any(c in message for c in ['http://', 'https://', ' | ']):
                return self.SkipReason.SKIP_TECHNICAL.value
            if actual_command_lines >= 3:
                return self.SkipReason.SKIP_TECHNICAL.value
        
        # Pattern 5: High path/URL density (dots and slashes suggesting file paths or URLs)
        if msg_len > 30:
            slash_count = message.count('/') + message.count('\\')
            dot_count = message.count('.')
            path_chars = slash_count + dot_count
            if path_chars > 10 and (path_chars / msg_len) > 0.15:
                return self.SkipReason.SKIP_TECHNICAL.value
        
        # Pattern 6: Markup character density (structured data)
        markup_chars = sum(message.count(c) for c in '{}[]<>')
        if markup_chars >= 6:
            if markup_chars / msg_len > 0.10:
                return self.SkipReason.SKIP_TECHNICAL.value
            # Special check for JSON-like structures (many nested braces)
            # Even with low density, if we have lots of curly braces, it's likely JSON
            curly_count = message.count('{') + message.count('}')
            if curly_count >= 10:
                return self.SkipReason.SKIP_TECHNICAL.value
        
        # Pattern 7: Structured nested content with colons (key: value patterns)
        line_count = message.count('\n')
        if line_count >= 8:  # At least 8 lines
            lines = message.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            if non_empty_lines:
                # Count lines with colon patterns (key: value or similar)
                colon_lines = sum(1 for line in non_empty_lines if ':' in line and not line.strip().startswith('#'))
                indented_lines = sum(1 for line in non_empty_lines if line.startswith((' ', '\t')))
                
                # If most lines have colons and indentation, it's structured data
                if (colon_lines / len(non_empty_lines) > 0.4 and 
                    indented_lines / len(non_empty_lines) > 0.5):
                    return self.SkipReason.SKIP_TECHNICAL.value
        
        # Pattern 8: Highly structured multi-line content (require markup chars for technical confidence)
        if line_count > 15:
            lines = message.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            if non_empty_lines:
                markup_in_lines = sum(1 for line in non_empty_lines if any(c in line for c in '{}[]<>'))
                structured_lines = sum(1 for line in non_empty_lines if line.startswith((' ', '\t')))
                
                # Require high markup presence or indented structure with technical keywords
                if markup_in_lines / len(non_empty_lines) > 0.3:
                    return self.SkipReason.SKIP_TECHNICAL.value
                elif structured_lines / len(non_empty_lines) > 0.6:
                    technical_keywords = ['function', 'class', 'import', 'return', 'const', 'var', 'let', 'def']
                    if any(keyword in message.lower() for keyword in technical_keywords):
                        return self.SkipReason.SKIP_TECHNICAL.value
        
        # Pattern 9: Code-like indentation pattern (require code indicators to avoid false positives from bullet lists)
        if line_count >= 3:
            lines = message.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            if non_empty_lines:
                indented_lines = sum(1 for line in non_empty_lines if line[0] in (' ', '\t'))
                if indented_lines / len(non_empty_lines) > 0.5:
                    code_indicators = ['def ', 'class ', 'function ', 'return ', 'import ', 'const ', 'let ', 'var ', 'public ', 'private ']
                    if any(indicator in message.lower() for indicator in code_indicators):
                        return self.SkipReason.SKIP_TECHNICAL.value
        
        # Pattern 10: Very high special character ratio (encoded data, technical output)
        if msg_len > 50:
            special_chars = sum(1 for c in message if not c.isalnum() and not c.isspace())
            special_ratio = special_chars / msg_len
            if special_ratio > 0.35:
                alphanumeric = sum(1 for c in message if c.isalnum())
                if alphanumeric / msg_len < 0.50:
                    return self.SkipReason.SKIP_TECHNICAL.value
        
        return None

    def detect_skip_reason(self, message: str, max_message_chars: int = Constants.MAX_MESSAGE_CHARS) -> Optional[str]:
        """
        Detect if a message should be skipped using two-stage detection:
        1. Fast-path structural patterns (~95% confidence)
        2. Semantic classification (for remaining cases)
        
        Returns:
            Skip reason string if content should be skipped, None otherwise
        """
        size_issue = self.validate_message_size(message, max_message_chars)
        if size_issue:
            return size_issue
        
        fast_skip = self._fast_path_skip_detection(message)
        if fast_skip:
            logger.info(f"Fast-path skip: {fast_skip}")
            return fast_skip
        
        if self._reference_embeddings is None:
            logger.warning("SkipDetector reference embeddings not initialized, allowing message through")
            return None
        
        try:
            from sentence_transformers import util
            
            message_embedding = self.embedding_model.encode(
                message.strip(),
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            conversational_similarities = util.cos_sim(
                message_embedding, 
                self._reference_embeddings['conversational']
            )[0]
            max_conversational_similarity = float(conversational_similarities.max())
            
            skip_categories = [
                ('instruction', self.SkipReason.SKIP_INSTRUCTION, self.INSTRUCTION_CATEGORY_DESCRIPTIONS),
                ('translation', self.SkipReason.SKIP_TRANSLATION, self.EXPLICIT_TRANSLATION_CATEGORY_DESCRIPTIONS),
                ('grammar', self.SkipReason.SKIP_GRAMMAR_PROOFREAD, self.GRAMMAR_PROOFREADING_CATEGORY_DESCRIPTIONS),
                ('technical', self.SkipReason.SKIP_TECHNICAL, self.TECHNICAL_CATEGORY_DESCRIPTIONS),
                ('pure_math', self.SkipReason.SKIP_PURE_MATH, self.PURE_MATH_CALCULATION_CATEGORY_DESCRIPTIONS),
            ]
            
            for cat_key, skip_reason, descriptions in skip_categories:
                similarities = util.cos_sim(
                    message_embedding, 
                    self._reference_embeddings[cat_key]
                )[0]
                max_similarity = float(similarities.max())
                
                if max_similarity > Constants.SKIP_DETECTION_SIMILARITY_THRESHOLD:
                    margin = max_similarity - max_conversational_similarity
                    
                    if margin > Constants.SKIP_DETECTION_CONFIDENT_MARGIN:
                        logger.info(f"Skipping message - {skip_reason.value} ({cat_key}: {max_similarity:.3f}, conv: {max_conversational_similarity:.3f}, margin: {margin:.3f})")
                        return skip_reason.value
                    
                    if margin > Constants.SKIP_DETECTION_MARGIN:                        
                        logger.info(f"Skipping message - {skip_reason.value} ({cat_key}: {max_similarity:.3f}, conv: {max_conversational_similarity:.3f}, margin: {margin:.3f})")
                        return skip_reason.value
            
            return None
            
        except Exception as e:
            logger.error(f"Error in semantic skip detection: {e}")
            return None


class LLMRerankingService:
    """Language-agnostic LLM-based memory reranking service."""

    def __init__(self, memory_system):
        self.memory_system = memory_system

    def _should_use_llm_reranking(self, memories: List[Dict]) -> Tuple[bool, str]:
        if not self.memory_system.valves.enable_llm_reranking:
            return False, "LLM reranking disabled"

        llm_trigger_threshold = int(self.memory_system.valves.max_memories_returned * self.memory_system.valves.llm_reranking_trigger_multiplier)
        if len(memories) > llm_trigger_threshold:
            return True, f"{len(memories)} candidate memories exceed {llm_trigger_threshold} threshold"            

        return False, f"{len(memories)} candidate memories within threshold of {llm_trigger_threshold}"

    async def _llm_select_memories(self, user_message: str, candidate_memories: List[Dict], max_count: int, emitter: Optional[Callable] = None) -> List[Dict]:
        """Use LLM to select most relevant memories."""
        memory_lines = self.memory_system._format_memories_for_llm(candidate_memories)
        memory_context = "\n".join(memory_lines)

        user_prompt = f"""CURRENT DATE/TIME: {self.memory_system.format_current_datetime()}

USER MESSAGE: {user_message}

CANDIDATE MEMORIES:
{memory_context}"""

        try:
            response = await self.memory_system._query_llm(Prompts.MEMORY_RERANKING, user_prompt, response_model=Models.MemoryRerankingResponse)

            selected_ids = response.ids

            selected_memories = []
            for memory in candidate_memories:
                if memory["id"] in selected_ids and len(selected_memories) < max_count:
                    selected_memories.append(memory)

            logger.info(f"🧠 LLM selected {len(selected_memories)} out of {len(candidate_memories)} candidates")
            
            return selected_memories

        except Exception as e:
            logger.warning(f"🤖 LLM reranking failed during memory relevance analysis: {str(e)}")            
            return candidate_memories

    async def rerank_memories(
        self, user_message: str, candidate_memories: List[Dict], emitter: Optional[Callable] = None
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        start_time = time.time()
        max_injection = self.memory_system.valves.max_memories_returned

        should_use_llm, decision_reason = self._should_use_llm_reranking(candidate_memories)

        analysis_info = {"llm_decision": should_use_llm, "decision_reason": decision_reason, "candidate_count": len(candidate_memories)}

        if should_use_llm:
            extended_count = int(self.memory_system.valves.max_memories_returned * Constants.EXTENDED_MAX_MEMORY_MULTIPLIER)
            llm_candidates = candidate_memories[:extended_count]
            await self.memory_system._emit_status(
                emitter, f"🤖 LLM Analyzing {len(llm_candidates)} Memories for Relevance", done=False
            )
            logger.info(f"Using LLM reranking: {decision_reason}")

            selected_memories = await self._llm_select_memories(user_message, llm_candidates, max_injection, emitter)
            
            if not selected_memories:
                logger.info("📭 No relevant memories after LLM analysis")
                await self.memory_system._emit_status(emitter, f"📭 No Relevant Memories After LLM Analysis", done=True)
                return selected_memories, analysis_info
        else:
            logger.info(f"Skipping LLM reranking: {decision_reason}")
            selected_memories = candidate_memories[:max_injection]

        duration = time.time() - start_time
        duration_text = f" in {duration:.2f}s" if duration >= 0.01 else ""
        await self.memory_system._emit_status(emitter, f"🎯 Memory Retrieval Complete{duration_text}", done=True)
        logger.info(f"🎯 Memory Retrieval Complete{duration_text}")
        return selected_memories, analysis_info


class LLMConsolidationService:
    """Language-agnostic LLM-based memory consolidation service."""

    def __init__(self, memory_system):
        self.memory_system = memory_system

    async def collect_consolidation_candidates(
        self, user_message: str, user_id: str, cached_similarities: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Collect candidate memories for consolidation analysis using cached or computed similarities."""
        if cached_similarities:
            consolidation_threshold = self.memory_system._get_retrieval_threshold(is_consolidation=True)
            candidates = [mem for mem in cached_similarities if mem["relevance"] >= consolidation_threshold]
            
            max_consolidation_memories = int(self.memory_system.valves.max_memories_returned * Constants.EXTENDED_MAX_MEMORY_MULTIPLIER)
            candidates = candidates[:max_consolidation_memories]

            logger.info(f"🎯 Found {len(candidates)} candidate memories for consolidation (threshold: {consolidation_threshold:.3f}, max: {max_consolidation_memories})")

            self.memory_system._log_retrieved_memories(candidates, "consolidation")
            return candidates

        try:
            user_memories = await self.memory_system._get_user_memories(user_id)
        except asyncio.TimeoutError:
            raise TimeoutError(f"⏱️ Memory retrieval timed out after {Constants.DATABASE_OPERATION_TIMEOUT_SEC}s")
        except Exception as e:
            logger.error(f"💾 Failed to retrieve user memories from database: {str(e)}")
            return []

        if not user_memories:
            logger.info("💭 No existing memories found for consolidation")
            return []
        else:
            logger.info(f"🚀 Reusing cached user memories for consolidation: {len(user_memories)} memories")

        try:
            all_similarities, _, _ = await self.memory_system._compute_similarities(user_message, user_id, user_memories)
        except Exception as e:
            logger.error(f"🔍 Failed to compute memory similarities for retrieval: {str(e)}")
            return []

        if all_similarities:
            consolidation_threshold = self.memory_system._get_retrieval_threshold(is_consolidation=True)
            candidates = [mem for mem in all_similarities if mem["relevance"] >= consolidation_threshold]
            
            max_consolidation_memories = int(self.memory_system.valves.max_memories_returned * Constants.EXTENDED_MAX_MEMORY_MULTIPLIER)
            candidates = candidates[:max_consolidation_memories]
            
            threshold_info = f"{consolidation_threshold:.3f} (max: {max_consolidation_memories})"
        else:
            candidates = []
            threshold_info = 'N/A'

        logger.info(f"🎯 Found {len(candidates)} candidate memories for consolidation (threshold: {threshold_info})")

        self.memory_system._log_retrieved_memories(candidates, "consolidation")

        return candidates

    async def generate_consolidation_plan(self, user_message: str, candidate_memories: List[Dict[str, Any]], emitter: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Generate consolidation plan using LLM with clear system/user prompt separation."""
        if candidate_memories:
            memory_lines = self.memory_system._format_memories_for_llm(candidate_memories)
            memory_context = f"EXISTING MEMORIES FOR CONSOLIDATION:\n{chr(10).join(memory_lines)}\n\n"
        else:
            memory_context = "EXISTING MEMORIES FOR CONSOLIDATION:\n[]\n\nNote: No existing memories found - Focus on extracting new memories from the user message below.\n\n"

        user_prompt = f"""CURRENT DATE/TIME: {self.memory_system.format_current_datetime()}

{memory_context}USER MESSAGE: {user_message}"""

        try:
            response = await asyncio.wait_for(
                self.memory_system._query_llm(Prompts.MEMORY_CONSOLIDATION, user_prompt, response_model=Models.ConsolidationResponse),
                timeout=Constants.LLM_CONSOLIDATION_TIMEOUT_SEC,
            )
        except Exception as e:
            logger.warning(f"🤖 LLM consolidation failed during memory processing: {str(e)}")
            await self.memory_system._emit_status(emitter, f"⚠️ Memory Consolidation Failed", done=True)
            return []

        operations = response.ops
        existing_memory_ids = {memory["id"] for memory in candidate_memories}

        total_operations = len(operations)
        delete_operations = [op for op in operations if op.operation == Models.MemoryOperationType.DELETE]
        delete_ratio = len(delete_operations) / total_operations if total_operations > 0 else 0

        if delete_ratio > Constants.MAX_DELETE_OPERATIONS_RATIO and total_operations >= Constants.MIN_OPS_FOR_DELETE_RATIO_CHECK:
            logger.warning(
                f"⚠️ Consolidation safety: {len(delete_operations)}/{total_operations} operations are deletions ({delete_ratio*100:.1f}%) - rejecting plan"
            )
            return []

        valid_operations = [op.model_dump() for op in operations if op.validate_operation(existing_memory_ids)]

        if valid_operations:
            create_count = sum(1 for op in valid_operations if op.get("operation") == Models.MemoryOperationType.CREATE.value)
            update_count = sum(1 for op in valid_operations if op.get("operation") == Models.MemoryOperationType.UPDATE.value)
            delete_count = sum(1 for op in valid_operations if op.get("operation") == Models.MemoryOperationType.DELETE.value)

            operation_details = self.memory_system._build_operation_details(create_count, update_count, delete_count)

            logger.info(f"🎯 Planned {len(valid_operations)} memory operations: {', '.join(operation_details)}")
        else:
            logger.info("🎯 No valid memory operations planned")

        return valid_operations

    async def execute_memory_operations(self, operations: List[Dict[str, Any]], user_id: str, emitter: Optional[Callable] = None) -> Tuple[int, int, int, int]:
        """Execute consolidation operations with simplified tracking."""
        if not operations or not user_id:
            return 0, 0, 0, 0

        try:
            user = await asyncio.wait_for(asyncio.to_thread(Users.get_user_by_id, user_id), timeout=Constants.DATABASE_OPERATION_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            raise TimeoutError(f"⏱️ User lookup timed out after {Constants.DATABASE_OPERATION_TIMEOUT_SEC}s")
        except Exception as e:
            raise RuntimeError(f"👤 User lookup failed: {str(e)}")

        if not user:
            raise ValueError(f"👤 User not found for consolidation: {user_id}")

        created_count = updated_count = deleted_count = failed_count = 0

        operations_by_type = {"CREATE": [], "UPDATE": [], "DELETE": []}
        for operation_data in operations:
            try:
                operation = Models.MemoryOperation(**operation_data)
                operations_by_type[operation.operation.value].append(operation)
            except Exception as e:
                failed_count += 1
                operation_type = operation_data.get("operation", Models.OperationResult.UNSUPPORTED.value)
                content_preview = ""
                if "content" in operation_data:
                    content = operation_data.get("content", "")
                    content_preview = f" - Content: {self.memory_system._truncate_content(content, Constants.CONTENT_PREVIEW_LENGTH)}"
                elif "id" in operation_data:
                    content_preview = f" - ID: {operation_data['id']}"
                error_message = f"Failed {operation_type} operation{content_preview}: {str(e)}"
                logger.error(error_message)

        memory_contents_for_deletion = {}
        if operations_by_type["DELETE"]:
            try:
                user_memories = await self.memory_system._get_user_memories(user_id)
                memory_contents_for_deletion = {str(mem.id): mem.content for mem in user_memories}
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch memories for DELETE preview: {str(e)}")

        for operation_type, ops in operations_by_type.items():
            if not ops:
                continue

            batch_tasks = []
            for operation in ops:
                task = self.memory_system._execute_single_operation(operation, user)
                batch_tasks.append(task)

            try:
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                for idx, result in enumerate(results):
                    operation = ops[idx]
                    
                    if isinstance(result, Exception):
                        failed_count += 1
                        await self.memory_system._emit_status(emitter, f"❌ Failed {operation_type}", done=False)
                    elif result == Models.MemoryOperationType.CREATE.value:
                        created_count += 1
                        content_preview = self.memory_system._truncate_content(operation.content)
                        await self.memory_system._emit_status(emitter, f"📝 Created: {content_preview}", done=False)
                    elif result == Models.MemoryOperationType.UPDATE.value:
                        updated_count += 1
                        content_preview = self.memory_system._truncate_content(operation.content)
                        await self.memory_system._emit_status(emitter, f"✏️ Updated: {content_preview}", done=False)
                    elif result == Models.MemoryOperationType.DELETE.value:
                        deleted_count += 1
                        content_preview = memory_contents_for_deletion.get(operation.id, operation.id)
                        if content_preview and content_preview != operation.id:
                            content_preview = self.memory_system._truncate_content(content_preview)
                        await self.memory_system._emit_status(emitter, f"🗑️ Deleted: {content_preview}", done=False)
                    elif result in [Models.OperationResult.FAILED.value, Models.OperationResult.UNSUPPORTED.value]:
                        failed_count += 1
                        await self.memory_system._emit_status(emitter, f"❌ Failed {operation_type}", done=False)
            except Exception as e:
                failed_count += len(ops)
                logger.error(f"❌ Batch {operation_type} operations failed during memory consolidation: {str(e)}")
                await self.memory_system._emit_status(emitter, f"❌ Batch {operation_type} Failed", done=False)

        total_executed = created_count + updated_count + deleted_count
        logger.info(
            f"✅ Memory processing completed {total_executed}/{len(operations)} operations (Created {created_count}, Updated {updated_count}, Deleted {deleted_count}, Failed {failed_count})"
        )

        if total_executed > 0:
            operation_details = self.memory_system._build_operation_details(created_count, updated_count, deleted_count)
            logger.info(f"🔄 Memory Operations: {', '.join(operation_details)}")
            await self.memory_system._manage_user_cache(user_id)

        return created_count, updated_count, deleted_count, failed_count

    async def run_consolidation_pipeline(
        self, user_message: str, user_id: str, emitter: Optional[Callable] = None, cached_similarities: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Complete consolidation pipeline with simplified flow."""
        start_time = time.time()
        try:
            if self.memory_system._shutdown_event.is_set():
                return

            candidates = await self.collect_consolidation_candidates(user_message, user_id, cached_similarities)
            if self.memory_system._shutdown_event.is_set():
                return

            operations = await self.generate_consolidation_plan(user_message, candidates, emitter)
            if self.memory_system._shutdown_event.is_set():
                return

            if operations:
                created_count, updated_count, deleted_count, failed_count = await self.execute_memory_operations(operations, user_id, emitter)
                
                duration = time.time() - start_time
                logger.info(f"💾 Memory Consolidation Complete In {duration:.2f}s")
                
                total_operations = created_count + updated_count + deleted_count
                if total_operations > 0 or failed_count > 0:
                    await self.memory_system._emit_status(emitter, f"💾 Memory Consolidation Complete in {duration:.2f}s", done=False)
                    
                    operation_details = self.memory_system._build_operation_details(created_count, updated_count, deleted_count)
                    memory_word = "Memory" if total_operations == 1 else "Memories"
                    operations_summary = f"{', '.join(operation_details)} {memory_word}"
                    
                    if failed_count > 0:
                        operations_summary += f" (❌ {failed_count} Failed)"
                    
                    await self.memory_system._emit_status(emitter, operations_summary, done=True)

        except Exception as e:
            duration = time.time() - start_time
            raise RuntimeError(f"❌ Memory consolidation failed after {duration:.2f}s: {str(e)}")


class Filter:
    """Enhanced multi-model embedding and memory filter with LRU caching."""

    __current_event_emitter__: Callable[[dict], Any]
    __user__: Dict[str, Any]
    __model__: str
    __request__: Request

    class Valves(BaseModel):
        """Configuration valves for the Memory System."""

        model: str = Field(default=Constants.DEFAULT_LLM_MODEL, description="Model name for LLM operations")
        embedding_model: str = Field(default=Constants.DEFAULT_EMBEDDING_MODEL, description="Sentence transformer model for embeddings")
        max_memories_returned: int = Field(default=Constants.MAX_MEMORIES_PER_RETRIEVAL, description="Maximum number of memories to return in context")
        max_message_chars: int = Field(default=Constants.MAX_MESSAGE_CHARS, description="Maximum user message length before skipping memory operations")
        semantic_retrieval_threshold: float = Field(default=Constants.SEMANTIC_RETRIEVAL_THRESHOLD, description="Minimum similarity threshold for memory retrieval")
        relaxed_semantic_threshold_multiplier: float = Field(default=Constants.RELAXED_SEMANTIC_THRESHOLD_MULTIPLIER, description="Adjusts similarity threshold for memory consolidation (lower = more candidates)")
        enable_llm_reranking: bool = Field(default=True, description="Enable LLM-based memory reranking for improved contextual selection")
        llm_reranking_trigger_multiplier: float = Field(default=Constants.LLM_RERANKING_TRIGGER_MULTIPLIER, description="Controls when LLM reranking activates (lower = more aggressive)")

    def __init__(self):
        """Initialize the Memory System filter with production validation."""
        global _SHARED_MODEL_CACHE
        
        self.valves = self.Valves()
        self._validate_system_configuration()

        self._cache_manager = UnifiedCacheManager(Constants.MAX_CACHE_ENTRIES_PER_TYPE, Constants.MAX_CONCURRENT_USER_CACHES)
        self._background_tasks: set = set()
        self._shutdown_event = asyncio.Event()

        model_key = self.valves.embedding_model
        
        if model_key in _SHARED_MODEL_CACHE:
            logger.info(f"♻️ Reusing cached embedding model: {model_key}")
            self._model = _SHARED_MODEL_CACHE[model_key]["model"]
            self._skip_detector = _SHARED_MODEL_CACHE[model_key]["skip_detector"]
        else:
            logger.info(f"🤖 Loading embedding model: {model_key} (cache has {len(_SHARED_MODEL_CACHE)} models)")
            self._model = SentenceTransformer(self.valves.embedding_model, device="cpu", trust_remote_code=True)
            self._skip_detector = SkipDetector(self._model)
            _SHARED_MODEL_CACHE[model_key] = {
                "model": self._model,
                "skip_detector": self._skip_detector
            }
            logger.info(f"✅ Embedding model and skip detector initialized and cached")

        self._llm_reranking_service = LLMRerankingService(self)
        self._llm_consolidation_service = LLMConsolidationService(self)

    def _set_pipeline_context(self, __event_emitter__: Optional[Callable] = None, __user__: Optional[Dict[str, Any]] = None, 
                             __model__: Optional[str] = None, __request__: Optional[Request] = None) -> None:
        """Set pipeline context parameters to avoid duplication in inlet/outlet methods."""
        if __event_emitter__:
            self.__current_event_emitter__ = __event_emitter__
        if __user__:
            self.__user__ = __user__
        if __model__:
            self.__model__ = __model__
        if __request__:
            self.__request__ = __request__

    def _truncate_content(self, content: str, max_length: Optional[int] = None) -> str:
        """Truncate content with ellipsis if needed."""
        if max_length is None:
            max_length = Constants.CONTENT_PREVIEW_LENGTH
        return content[:max_length] + "..." if len(content) > max_length else content

    def _get_retrieval_threshold(self, is_consolidation: bool = False) -> float:
        """Calculate retrieval threshold for semantic similarity filtering."""
        if is_consolidation:
            return self.valves.semantic_retrieval_threshold * self.valves.relaxed_semantic_threshold_multiplier
        return self.valves.semantic_retrieval_threshold

    def _extract_text_from_content(self, content) -> str:
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return " ".join(text_parts)
        elif isinstance(content, dict) and "text" in content:
            return content["text"]
        return ""

    def _validate_system_configuration(self) -> None:
        """Validate configuration and fail if invalid."""
        if not self.valves.model or not self.valves.model.strip():
            raise ValueError("🤖 Model not specified")

        if self.valves.max_memories_returned <= 0:
            raise ValueError(f"📊 Invalid max memories returned: {self.valves.max_memories_returned}")

        if not (0.0 <= self.valves.semantic_retrieval_threshold <= 1.0):
            raise ValueError(f"🎯 Invalid semantic retrieval threshold: {self.valves.semantic_retrieval_threshold} (must be 0.0-1.0)")

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

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding vector."""
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def _generate_embeddings_sync(self, model, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Synchronous embedding generation for single text or batch."""
        is_single = isinstance(texts, str)
        input_texts = [texts] if is_single else texts

        embeddings = model.encode(input_texts, convert_to_numpy=True, show_progress_bar=False)
        normalized = [self._normalize_embedding(emb) for emb in embeddings]

        return normalized[0] if is_single else normalized

    async def _generate_embeddings(self, texts: Union[str, List[str]], user_id: str) -> Union[np.ndarray, List[np.ndarray]]:
        """Unified embedding generation for single text or batch with optimized caching."""
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts

        if not text_list:
            if is_single:
                raise ValueError("📏 Empty text provided for embedding generation")
            return []

        result_embeddings = []
        uncached_texts = []
        uncached_indices = []
        uncached_hashes = []

        for i, text in enumerate(text_list):
            if not text or len(str(text).strip()) < Constants.MIN_MESSAGE_CHARS:
                if is_single:
                    raise ValueError("📏 Text too short for embedding generation")
                result_embeddings.append(None)
                continue

            text_hash = self._compute_text_hash(str(text))
            cached = await self._get_embedding_cache(user_id, text_hash)

            if cached is not None:
                result_embeddings.append(cached)
            else:
                result_embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
                uncached_hashes.append(text_hash)

        if uncached_texts:
            loop = asyncio.get_event_loop()
            new_embeddings = await loop.run_in_executor(None, self._generate_embeddings_sync, self._model, uncached_texts)

            for j, embedding in enumerate(new_embeddings):
                original_idx = uncached_indices[j]
                text_hash = uncached_hashes[j]
                await self._put_embedding_cache(user_id, text_hash, embedding)
                result_embeddings[original_idx] = embedding

        if is_single:
            logger.info("📥 User message embedding: cache hit" if not uncached_texts else "💾 User message embedding: generated and cached")
            return result_embeddings[0]
        else:
            valid_count = sum(1 for emb in result_embeddings if emb is not None)
            logger.info(
                f"🚀 Batch embedding: {len(text_list) - len(uncached_texts)} cached, {len(uncached_texts)} new, {valid_count}/{len(text_list)} valid"
            )
            return result_embeddings

    def _should_skip_memory_operations(self, user_message: str) -> Tuple[bool, str]:
        if self._skip_detector is None:
            raise RuntimeError("🤖 Skip detector not initialized")
        
        skip_reason = self._skip_detector.detect_skip_reason(user_message, self.valves.max_message_chars)
        if skip_reason:
            status_key = SkipDetector.SkipReason(skip_reason)
            return True, SkipDetector.STATUS_MESSAGES[status_key]
        return False, ""

    def _process_user_message(self, body: Dict[str, Any]) -> Tuple[Optional[str], bool, str]:
        """Extract user message and determine if memory operations should be skipped."""
        if not body or "messages" not in body or not isinstance(body["messages"], list):
            return None, True, SkipDetector.STATUS_MESSAGES[SkipDetector.SkipReason.SKIP_SIZE]

        messages = body["messages"]
        user_message = None

        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue

            content = message.get("content", "")
            user_message = self._extract_text_from_content(content)

            if user_message:
                break

        if not user_message or not user_message.strip():
            return None, True, SkipDetector.STATUS_MESSAGES[SkipDetector.SkipReason.SKIP_SIZE]

        should_skip, skip_reason = self._should_skip_memory_operations(user_message)
        return user_message, should_skip, skip_reason

    async def _get_user_memories(self, user_id: str, timeout: Optional[float] = None) -> List:
        """Get user memories with timeout handling."""
        if timeout is None:
            timeout = Constants.DATABASE_OPERATION_TIMEOUT_SEC
        try:
            return await asyncio.wait_for(asyncio.to_thread(Memories.get_memories_by_user_id, user_id), timeout=timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"⏱️ Memory retrieval timed out after {timeout}s")
        except Exception as e:
            raise RuntimeError(f"💾 Memory retrieval failed: {str(e)}")

    def _log_retrieved_memories(self, memories: List[Dict[str, Any]], context_type: str = "semantic") -> None:
        """Log retrieved memories with concise formatting showing key statistics and semantic values."""
        if not memories:
            return

        scores = [memory["relevance"] for memory in memories]

        if not scores:
            return

        top_score = max(scores)
        lowest_score = min(scores)
        median_score = sorted(scores)[len(scores) // 2]

        context_label = "📊 Consolidation candidate memories" if context_type == "consolidation" else "📊 Retrieved memories"
        max_scores_to_show = int(self.valves.max_memories_returned * Constants.EXTENDED_MAX_MEMORY_MULTIPLIER)
        scores_str = ", ".join([f"{score:.3f}" for score in scores[:max_scores_to_show]])
        suffix = "..." if len(scores) > max_scores_to_show else ""

        logger.info(f"{context_label}: {len(memories)} memories | Top: {top_score:.3f} | Median: {median_score:.3f} | Lowest: {lowest_score:.3f}")
        logger.info(f"Scores: [{scores_str}{suffix}]")

    def _build_operation_details(self, created_count: int, updated_count: int, deleted_count: int) -> List[str]:
        """Build operation details list with consistent formatting."""
        operation_details = []

        operations = [(created_count, "📝 Created"), (updated_count, "✏️ Updated"), (deleted_count, "🗑️ Deleted")]

        for count, label in operations:
            if count > 0:
                operation_details.append(f"{label} {count}")

        return operation_details

    def _cache_key(self, cache_type: str, user_id: str, content: Optional[str] = None) -> str:
        """Unified cache key generation for all cache types."""
        if content:
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()[: Constants.CACHE_KEY_HASH_PREFIX_LENGTH]
            return f"{cache_type}_{user_id}:{content_hash}"
        return f"{cache_type}_{user_id}"

    def format_current_datetime(self) -> str:
        try:
            now = datetime.now(timezone.utc)
            return now.strftime("%A %B %d %Y at %H:%M:%S UTC")
        except Exception as e:
            raise RuntimeError(f"📅 Failed to format datetime: {str(e)}")

    def _format_memories_for_llm(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Format memories for LLM consumption with hybrid format and human-readable timestamps."""
        memory_lines = []
        for memory in memories:
            line = f"[{memory['id']}] {memory['content']}"
            record_date = memory.get("updated_at") or memory.get("created_at")
            if record_date:
                try:
                    if isinstance(record_date, str):
                        parsed_date = datetime.fromisoformat(record_date.replace('Z', '+00:00'))
                    else:
                        parsed_date = record_date
                    formatted_date = parsed_date.strftime("%b %d %Y")
                    line += f" [noted at {formatted_date}]"
                except Exception as e:
                    logger.warning(f"Failed to format date {record_date}: {str(e)}")
                    line += f" [noted at {record_date}]"
            memory_lines.append(line)
        return memory_lines

    async def _emit_status(self, emitter: Optional[Callable], description: str, done: bool = True) -> None:
        """Emit status messages for memory operations."""
        if not emitter:
            return

        payload = {"type": "status", "data": {"description": description, "done": done}}

        try:
            result = emitter(payload)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass

    async def _retrieve_relevant_memories(
        self,
        user_message: str,
        user_id: str,
        user_memories: Optional[List] = None,
        emitter: Optional[Callable] = None,
        cached_similarities: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Retrieve memories for injection using similarity computation with optional LLM reranking."""
        if cached_similarities is not None:
            memories = [m for m in cached_similarities if m.get("relevance", 0) >= self.valves.semantic_retrieval_threshold]
            logger.info(f"🔍 Using cached similarities for {len(memories)} candidate memories")
            final_memories, reranking_info = await self._llm_reranking_service.rerank_memories(user_message, memories, emitter)
            self._log_retrieved_memories(final_memories, "semantic")
            return {
                "memories": final_memories,
                "threshold": self.valves.semantic_retrieval_threshold,
                "all_similarities": cached_similarities,
                "reranking_info": reranking_info,
            }

        if user_memories is None:
            user_memories = await self._get_user_memories(user_id)

        if not user_memories:
            logger.info("📭 No memories found for user")
            await self._emit_status(emitter, "📭 No Memories Found", done=True)
            return {"memories": [], "threshold": None}

        memories, threshold, all_similarities = await self._compute_similarities(user_message, user_id, user_memories)

        if memories:
            final_memories, reranking_info = await self._llm_reranking_service.rerank_memories(user_message, memories, emitter)
        else:
            logger.info("📭 No relevant memories found above similarity threshold")
            await self._emit_status(emitter, "📭 No Relevant Memories Found", done=True)
            final_memories = memories
            reranking_info = {"llm_decision": False, "decision_reason": "no_candidates"}

        self._log_retrieved_memories(final_memories, "semantic")

        return {"memories": final_memories, "threshold": threshold, "all_similarities": all_similarities, "reranking_info": reranking_info}

    async def _add_memory_context(
        self, body: Dict[str, Any], memories: Optional[List[Dict[str, Any]]] = None, user_id: Optional[str] = None, emitter: Optional[Callable] = None
    ) -> None:
        """Add memory context to request body with simplified logic."""
        if not body or "messages" not in body or not body["messages"]:
            logger.warning("⚠️ Invalid request body or no messages found")
            return

        content_parts = [f"Current Date/Time: {self.format_current_datetime()}"]

        memory_count = 0
        if memories and user_id:
            memory_count = len(memories)
            memory_header = f"CONTEXT: The following {'fact' if memory_count == 1 else 'facts'} about the user are provided for background only. Not all facts may be relevant to the current request."
            formatted_memories = []
            
            for idx, memory in enumerate(memories, 1):
                formatted_memory = f"- {' '.join(memory['content'].split())}"
                formatted_memories.append(formatted_memory)
                
                content_preview = self._truncate_content(memory['content'])
                await self._emit_status(emitter, f"💭 {idx}/{memory_count}: {content_preview}", done=False)
            
            memory_footer = "IMPORTANT: Do not mention or imply you received this list. These facts are for background context only."
            memory_context_block = f"{memory_header}\n{chr(10).join(formatted_memories)}\n\n{memory_footer}"
            content_parts.append(memory_context_block)

        memory_context = "\n\n".join(content_parts)

        system_index = next((i for i, msg in enumerate(body["messages"]) if msg.get("role") == "system"), None)

        if system_index is not None:
            body["messages"][system_index]["content"] = f"{body['messages'][system_index].get('content', '')}\n\n{memory_context}"
        else:
            body["messages"].insert(0, {"role": "system", "content": memory_context})
        
        if memories and user_id:
            description = f"🧠 Injected {memory_count} {'Memory' if memory_count == 1 else 'Memories'} to Context"
            await self._emit_status(emitter, description, done=True)

    def _build_memory_dict(self, memory, similarity: float) -> Dict[str, Any]:
        """Build memory dictionary with standardized timestamp conversion."""
        memory_dict = {"id": str(memory.id), "content": memory.content, "relevance": similarity}
        if hasattr(memory, "created_at") and memory.created_at:
            memory_dict["created_at"] = datetime.fromtimestamp(memory.created_at, tz=timezone.utc).isoformat()
        if hasattr(memory, "updated_at") and memory.updated_at:
            memory_dict["updated_at"] = datetime.fromtimestamp(memory.updated_at, tz=timezone.utc).isoformat()
        return memory_dict

    async def _compute_similarities(
        self, user_message: str, user_id: str, user_memories: List
    ) -> Tuple[List[Dict], float, List[Dict]]:
        """Compute similarity scores between user message and memories."""
        if not user_memories:
            return [], self.valves.semantic_retrieval_threshold, []

        query_embedding = await self._generate_embeddings(user_message, user_id)
        memory_contents = [memory.content for memory in user_memories]
        memory_embeddings = await self._generate_embeddings(memory_contents, user_id)

        if len(memory_embeddings) != len(user_memories):
            logger.error(f"🔢 Embedding generation failed: generated {len(memory_embeddings)} embeddings but expected {len(user_memories)} for user memories")
            return [], self.valves.semantic_retrieval_threshold, []

        similarity_scores = []
        memory_data = []

        for memory_index, memory in enumerate(user_memories):
            memory_embedding = memory_embeddings[memory_index]
            if memory_embedding is None:
                continue

            similarity = float(np.dot(query_embedding, memory_embedding))
            similarity_scores.append(similarity)
            memory_dict = self._build_memory_dict(memory, similarity)
            memory_data.append(memory_dict)

        if not similarity_scores:
            return [], self.valves.semantic_retrieval_threshold, []

        memory_data.sort(key=lambda x: x["relevance"], reverse=True)

        threshold = self.valves.semantic_retrieval_threshold
        filtered_memories = [m for m in memory_data if m["relevance"] >= threshold]        
        return filtered_memories, threshold, memory_data

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable] = None,
        __user__: Optional[Dict[str, Any]] = None,
        __model__: Optional[str] = None,
        __request__: Optional[Request] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Simplified inlet processing for memory retrieval and injection."""
        self._set_pipeline_context(__event_emitter__, __user__, __model__, __request__)

        user_id = __user__.get("id") if body and __user__ else None
        if not user_id:
            return body

        user_message, should_skip, skip_reason = self._process_user_message(body)

        if not user_message or should_skip:
            if __event_emitter__ and skip_reason:
                await self._emit_status(__event_emitter__, skip_reason, done=True)
            await self._add_memory_context(body, [], user_id, __event_emitter__)
            return body

        try:
            memory_cache_key = self._cache_key(self._cache_manager.MEMORY_CACHE, user_id)
            user_memories = await self._cache_manager.get(user_id, self._cache_manager.MEMORY_CACHE, memory_cache_key)

            if user_memories is None:
                user_memories = await self._get_user_memories(user_id)
                if user_memories:
                    await self._cache_manager.put(user_id, self._cache_manager.MEMORY_CACHE, memory_cache_key, user_memories)

            retrieval_result = await self._retrieve_relevant_memories(user_message, user_id, user_memories, __event_emitter__)
            memories = retrieval_result.get("memories", [])
            threshold = retrieval_result.get("threshold")
            all_similarities = retrieval_result.get("all_similarities", [])

            if all_similarities:
                cache_key = self._cache_key(self._cache_manager.RETRIEVAL_CACHE, user_id, user_message)
                await self._cache_manager.put(user_id, self._cache_manager.RETRIEVAL_CACHE, cache_key, all_similarities)

            await self._add_memory_context(body, memories, user_id, __event_emitter__)

        except Exception as e:
            raise RuntimeError(f"💾 Memory retrieval failed: {str(e)}")

        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Optional[Callable] = None,
        __user__: Optional[dict] = None,
        __model__: Optional[str] = None,
        __request__: Optional[Request] = None,
        **kwargs,
    ) -> dict:
        """Simplified outlet processing for background memory consolidation."""
        self._set_pipeline_context(__event_emitter__, __user__, __model__, __request__)

        user_id = __user__.get("id") if body and __user__ else None
        if not user_id:
            return body

        user_message, should_skip, skip_reason = self._process_user_message(body)

        if not user_message or should_skip:
            return body

        cache_key = self._cache_key(self._cache_manager.RETRIEVAL_CACHE, user_id, user_message)
        cached_similarities = await self._cache_manager.get(user_id, self._cache_manager.RETRIEVAL_CACHE, cache_key)

        task = asyncio.create_task(
            self._llm_consolidation_service.run_consolidation_pipeline(user_message, user_id, __event_emitter__, cached_similarities)
        )
        self._background_tasks.add(task)

        def safe_cleanup(t: asyncio.Task) -> None:
            try:
                self._background_tasks.discard(t)
                if t.exception() and not t.cancelled():
                    exception = t.exception()
                    logger.error(f"❌ Background memory consolidation task failed: {str(exception)}")
            except Exception as e:
                logger.error(f"❌ Failed to cleanup background memory task: {str(e)}")

        task.add_done_callback(safe_cleanup)

        return body

    async def shutdown(self) -> None:
        """Cleanup method to properly shutdown background tasks."""
        self._shutdown_event.set()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

        await self._cache_manager.clear_all_caches()

    async def _manage_user_cache(self, user_id: str, clear_first: bool = False) -> None:
        """Manage user cache - clear, invalidate, and refresh as needed."""
        start_time = time.time()
        try:
            if clear_first:
                total_removed = await self._cache_manager.clear_user_cache(user_id)
                logger.info(f"🧹 Cleared {total_removed} cache entries for user {user_id}")
            else:
                retrieval_cleared = await self._cache_manager.clear_user_cache(user_id, self._cache_manager.RETRIEVAL_CACHE)
                logger.info(f"🔄 Cleared {retrieval_cleared} retrieval cache entries for user {user_id}")

            user_memories = await self._get_user_memories(user_id)
            memory_cache_key = self._cache_key(self._cache_manager.MEMORY_CACHE, user_id)

            if not user_memories:
                await self._cache_manager.put(user_id, self._cache_manager.MEMORY_CACHE, memory_cache_key, [])
                logger.info("📭 No memories found for user")
                return

            await self._cache_manager.put(user_id, self._cache_manager.MEMORY_CACHE, memory_cache_key, user_memories)

            memory_contents = [
                memory.content
                for memory in user_memories
                if memory.content and len(memory.content.strip()) >= Constants.MIN_MESSAGE_CHARS
            ]

            if memory_contents:
                await self._generate_embeddings(memory_contents, user_id)
                duration = time.time() - start_time
                logger.info(f"� Cache updated with {len(memory_contents)} embeddings for user {user_id} in {duration:.2f}s")

        except Exception as e:
            raise RuntimeError(f"🧹 Failed to manage cache for user {user_id} after {(time.time() - start_time):.2f}s: {str(e)}")

    async def _execute_single_operation(self, operation: Models.MemoryOperation, user: Any) -> str:
        """Execute a single memory operation."""
        try:
            if operation.operation == Models.MemoryOperationType.CREATE:
                if not operation.content.strip():
                    logger.warning(f"⚠️ Skipping CREATE operation: empty content")
                    return Models.OperationResult.SKIPPED_EMPTY_CONTENT.value

                await asyncio.wait_for(
                    asyncio.to_thread(Memories.insert_new_memory, user.id, operation.content.strip()), timeout=Constants.DATABASE_OPERATION_TIMEOUT_SEC
                )
                return Models.MemoryOperationType.CREATE.value

            elif operation.operation == Models.MemoryOperationType.UPDATE:
                if not operation.id.strip():
                    logger.warning(f"⚠️ Skipping UPDATE operation: empty ID")
                    return Models.OperationResult.SKIPPED_EMPTY_ID.value
                if not operation.content.strip():
                    logger.warning(f"⚠️ Skipping UPDATE operation for {operation.id}: empty content")
                    return Models.OperationResult.SKIPPED_EMPTY_CONTENT.value

                await asyncio.wait_for(
                    asyncio.to_thread(Memories.update_memory_by_id_and_user_id, operation.id, user.id, operation.content.strip()),
                    timeout=Constants.DATABASE_OPERATION_TIMEOUT_SEC,
                )
                return Models.MemoryOperationType.UPDATE.value

            elif operation.operation == Models.MemoryOperationType.DELETE:
                if not operation.id.strip():
                    logger.warning(f"⚠️ Skipping DELETE operation: empty ID")
                    return Models.OperationResult.SKIPPED_EMPTY_ID.value

                await asyncio.wait_for(
                    asyncio.to_thread(Memories.delete_memory_by_id_and_user_id, operation.id, user.id), timeout=Constants.DATABASE_OPERATION_TIMEOUT_SEC
                )
                return Models.MemoryOperationType.DELETE.value
            else:
                logger.error(f"❓ Unsupported operation: {operation}")
                return Models.OperationResult.UNSUPPORTED.value

        except Exception as e:
            logger.error(f"💾 Database operation failed for {operation.operation.value}: {str(e)}")
            return Models.OperationResult.FAILED.value

    def _remove_refs_from_schema(self, schema: Dict[str, Any], schema_defs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Remove $ref references and ensure required fields for Azure OpenAI."""
        if not isinstance(schema, dict):
            return schema
        
        if '$ref' in schema:
            ref_path = schema['$ref']
            if ref_path.startswith('#/$defs/'):
                def_name = ref_path.split('/')[-1]
                if schema_defs and def_name in schema_defs:
                    return self._remove_refs_from_schema(schema_defs[def_name].copy(), schema_defs)
            return {'type': 'object'}
        
        result = {}
        for key, value in schema.items():
            if key == '$defs':
                continue
            elif isinstance(value, dict):
                result[key] = self._remove_refs_from_schema(value, schema_defs)
            elif isinstance(value, list):
                result[key] = [self._remove_refs_from_schema(item, schema_defs) if isinstance(item, dict) else item for item in value]
            else:
                result[key] = value
        
        if result.get('type') == 'object' and 'properties' in result:
            result['required'] = list(result['properties'].keys())
        
        return result

    async def _query_llm(self, system_prompt: str, user_prompt: str, response_model: Optional[BaseModel] = None) -> Union[str, BaseModel]:
        """Query OpenWebUI's internal model system with Pydantic model parsing."""
        if not hasattr(self, "__request__") or not hasattr(self, "__user__"):
            raise RuntimeError("🔧 Pipeline interface not properly initialized. __request__ and __user__ required.")

        model_to_use = self.valves.model if self.valves.model else self.__model__
        if not model_to_use:
            raise ValueError("🤖 No model specified for LLM operations")

        form_data = {
            "model": model_to_use,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],            
            "max_tokens": 4096,
            "stream": False,
        }

        if response_model:
            raw_schema = response_model.model_json_schema()
            schema_defs = raw_schema.get('$defs', {})
            schema = self._remove_refs_from_schema(raw_schema, schema_defs)
            schema['type'] = 'object'
            form_data["response_format"] = {"type": "json_schema", "json_schema": {"name": response_model.__name__, "strict": True, "schema": schema}}

        try:
            response = await asyncio.wait_for(
                generate_chat_completion(self.__request__, form_data, user=await asyncio.to_thread(Users.get_user_by_id, self.__user__["id"])),
                timeout=Constants.LLM_CONSOLIDATION_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"⏱️ LLM query timed out after {Constants.LLM_CONSOLIDATION_TIMEOUT_SEC}s")
        except Exception as e:
            raise RuntimeError(f"🤖 LLM query failed: {str(e)}")

        try:
            if hasattr(response, "body") and hasattr(getattr(response, "body", None), "decode"):
                body = getattr(response, "body")
                response_data = json.loads(body.decode("utf-8"))
            else:
                response_data = response
        except (json.JSONDecodeError, AttributeError) as e:
            raise RuntimeError(f"🔍 Failed to decode response body: {str(e)}")

        if isinstance(response_data, dict) and "choices" in response_data and isinstance(response_data["choices"], list) and len(response_data["choices"]) > 0:
            first_choice = response_data["choices"][0]
            if isinstance(first_choice, dict) and "message" in first_choice and isinstance(first_choice["message"], dict) and "content" in first_choice["message"]:
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
