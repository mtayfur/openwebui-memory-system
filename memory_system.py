"""
title: Memory System
description: A semantic memory management system for Open WebUI that consolidates, deduplicates, and retrieves personalized user memories using LLM operations.
version: 1.2.1
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

_SHARED_SKIP_DETECTOR_CACHE = {}
_SHARED_SKIP_DETECTOR_CACHE_LOCK = asyncio.Lock()


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
    MAX_CACHE_ENTRIES_PER_TYPE = 500  # Maximum cache entries per cache type
    MAX_CONCURRENT_USER_CACHES = 50  # Maximum concurrent user cache instances
    CACHE_KEY_HASH_PREFIX_LENGTH = 10  # Hash prefix length for cache keys

    # Retrieval & Similarity
    SEMANTIC_RETRIEVAL_THRESHOLD = 0.25  # Semantic similarity threshold for retrieval
    RELAXED_SEMANTIC_THRESHOLD_MULTIPLIER = 0.8  # Multiplier for relaxed similarity threshold in secondary operations
    EXTENDED_MAX_MEMORY_MULTIPLIER = 1.5  # Multiplier for expanding memory candidates in advanced operations
    LLM_RERANKING_TRIGGER_MULTIPLIER = 0.5  # Multiplier for LLM reranking trigger threshold

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

    MEMORY_CONSOLIDATION = f"""You are the Memory System Consolidator, a specialist in creating precise user memories.

## OBJECTIVE
Your goal is to build precise memories of the user's personal narrative with factual, temporal statements.

## AVAILABLE OPERATIONS
- CREATE: For new, personal facts. Must be semantically and temporally enhanced.
- UPDATE: To modify existing memories, including making facts historical with a date range.
- DELETE: For explicit user requests or to resolve contradictions.
- SKIP: When no new, personal information is provided.

## PROCESSING GUIDELINES
- Personal Facts Only: Store only significant facts with lasting relevance to the user's life and identity. Exclude transient situations, questions, general knowledge, casual mentions, or momentary states.
- **Filter for Intent:** You MUST SKIP if the user's primary intent is instructional, technical, or analytical, even if the message contains personal details. This includes requests to:
    - Rewrite, revise, translate, or proofread a block of text (e.g., "revise this review for me").
    - Answer a general knowledge, math, or technical question.
    - Explain a concept, perform a calculation, or act as a persona.
  **Only store facts when the user is *directly stating* them as part of a personal narrative, not when providing them as content for a task.**
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
    - Mandatory Semantic Enhancement: Enhance entities with descriptive categorical nouns for better retrieval.
    - Verify Nouns/Pronouns: Link pronouns (he, she, they) and nouns to specific entities.
    - First-Person Format: Write all memories in English from the user's perspective.

## DECISION FRAMEWORK
- Selectivity: Verify the user's *primary intent* is to state a direct, personally significant fact with lasting importance. If the intent is instructional, analytical, or a general question, SKIP. Never create duplicate memories. Skip momentary events or casual mentions. Be conservative with CREATE and UPDATE operations.
- Strategy: Strongly prioritize enriching existing memories over creating new ones. Analyze the message holistically to identify naturally connected facts (same person, event, or timeframe) and combine them into a unified, cohesive memory rather than fragmenting them. Each memory must be self-contained and **never** merge unrelated information.
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
Explanation: Dog memory enriched with related context (Emma, birthday gift, age 11) and temporal anchoring (September 2024). The instructional question ("What should I give her...?") is ignored as per the 'Filter for Intent' rule.

### Example 3
Message: "Can you recommend some good tapas restaurants in Barcelona? I moved here from Madrid last month."
Memories: [id:mem-005] I live in Madrid Spain [noted at June 12 2025]
Return: {{"ops": [{{"operation": "UPDATE", "id": "mem-005", "content": "I lived in Madrid Spain until August 2025"}}, {{"operation": "CREATE", "id": "", "content": "I moved to Barcelona Spain in August 2025"}}]}}
Explanation: Relocation is a significant life event. The request for recommendations is instructional and is ignored.

### Example 4
Message: "My wife Sofia and I just got married in August. What are some good honeymoon destinations?"
Memories: [id:mem-008] I am single [noted at January 5 2025]
Return: {{"ops": [{{"operation": "DELETE", "id": "mem-008", "content": ""}}, {{"operation": "CREATE", "id": "", "content": "I married Sofia in August 2025 and she is now my wife"}}]}}
Explanation: Marriage is an enduring life event. The instructional question ("What are some good honeymoon destinations?") is ignored.

### Example 5
Message: "¡Hola! Me mudé de Madrid a Barcelona el mes pasado y me casé con mi novia Sofía en agosto. ¿Me puedes recomendar un buen restaurante para celebrar?"
Memories: [id:mem-005] I live in Madrid Spain [noted at June 12 2025] [id:mem-006] I am dating Sofia [noted at February 10 2025] [id:mem-008] I am single [noted at January 5 2025]
Return: {{"ops": [{{"operation": "UPDATE", "id": "mem-005", "content": "I lived in Madrid Spain until August 2025"}}, {{"operation": "DELETE", "id": "mem-008", "content": ""}}, {{"operation": "UPDATE", "id": "mem-006", "content": "I moved to Barcelona Spain and married my girlfriend Sofia in August 2025, who is now my wife"}}]}}
Explanation: The user's move and marriage are significant, related life events. They are consolidated into a single memory. The request for a recommendation is ignored.

### Example 6
Message: "I'm feeling stressed about work this week and looking for some relaxation tips. I have a big presentation coming up on Friday."
Memories: []
Return: {{"ops": []}}
Explanation: Transient state (stress) and a request for information (relaxation tips). The primary intent is instructional/analytical, and the facts (presentation) are not significant, lasting personal narrative. Nothing to store.
"""

    MEMORY_RERANKING = f"""You are the Memory Relevance Analyzer.

## OBJECTIVE
Your goal is to analyze the user's message and select the most relevant memories to personalize the AI's response. Prioritize direct connections and supporting context.

## RELEVANCE CATEGORIES
- Direct: Memories explicitly about the query topic, people, or domain.
- Contextual: Personal info that affects response recommendations or understanding.
- Background: Situational context that provides useful personalization.

## SELECTION FRAMEWORK
- Prioritize Current Info: Give current facts higher relevance than historical ones unless the query is about the past or historical context directly informs the current situation.
- Hierarchy: Prioritize topic matches first (Direct), then context that enhances the response (Contextual), and finally general background (Background).
- Ordering: Order IDs by relevance, most relevant first.
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

    class StatusEmitLevel(str, Enum):
        """Verbosity levels for status message emission - selectable as dropdown with title case strings."""

        BASIC = "Basic"
        INTERMEDIATE = "Intermediate"
        DETAILED = "Detailed"

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

        operation: "Models.MemoryOperationType" = Field(description="Type of memory operation to perform")
        content: str = Field(description="Memory content (required for CREATE/UPDATE, empty for DELETE)")
        id: str = Field(description="Memory ID (empty for CREATE, required for UPDATE/DELETE)")

        def validate_operation(self, existing_memory_ids: Optional[set] = None) -> bool:
            """Validate the memory operation against existing memory IDs."""
            if existing_memory_ids is None:
                existing_memory_ids = set()

            if self.operation == Models.MemoryOperationType.CREATE:
                return True
            elif self.operation in [
                Models.MemoryOperationType.UPDATE,
                Models.MemoryOperationType.DELETE,
            ]:
                return self.id in existing_memory_ids
            return False

    class ConsolidationResponse(BaseModel):
        """Pydantic model for memory consolidation LLM response - object containing array of memory operations."""

        ops: List["Models.MemoryOperation"] = Field(default_factory=list, description="List of memory operations to execute")

    class MemoryRerankingResponse(BaseModel):
        """Pydantic model for memory reranking LLM response - object containing array of memory IDs."""

        ids: List[str] = Field(
            default_factory=list,
            description="List of memory IDs selected as most relevant for the user query",
        )


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
        self.SKIP_STATE_CACHE = "skip"

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
                    self.caches.popitem(last=False)
                self.caches[user_id] = {}

            user_cache = self.caches[user_id]

            if cache_type not in user_cache:
                user_cache[cache_type] = OrderedDict()

            type_cache = user_cache[cache_type]

            if key not in type_cache and len(type_cache) >= self.max_cache_size_per_type:
                type_cache.popitem(last=False)

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


class SkipDetector:
    """Binary content classifier: personal vs non-personal using semantic analysis."""

    NON_PERSONAL_CATEGORY_DESCRIPTIONS = [
        # --- Abstract Knowledge & Creative Tasks ---
        "General knowledge questions about **impersonal, academic, or abstract topics** like geography, world history, trivia, theoretical science, or definitions. 'What is the capital of France?', 'Who was the 1st president?', 'Explain quantum physics'.",
        "General knowledge explanations of concepts, mechanisms, or processes. 'Define photosynthesis', 'How does a combustion engine work?', 'Explain how blockchain technology operates', 'What is the theory of relativity?', 'Describe DNA replication'.",
        "Creative writing prompts, requests for jokes, poems, fictional stories, or content generation. 'Write a poem about a tree', 'Generate a story where...', 'Draft a marketing email for a fake product', 'Create a character backstory', 'Write a song'.",
        "Requests for generic recommendations, lists, outlines, or brainstorming on impersonal topics without personal context. 'Give me 10 ideas for a sci-fi movie', 'Brainstorm names for a tech company', 'Create an outline for an essay on Rome'.",
        "Requests for advice, suggestions, or recommendations where the PRIMARY INTENT is to get help or information, even if personal context is mentioned. 'What should I give my daughter for her birthday?', 'Can you recommend restaurants in my city?'.",
        "Seeking recommendations or help with personal decisions where the question is the focus, not stating facts. 'What are good honeymoon destinations?', 'Help me choose between job offers', 'What car should I buy for my commute?', 'Which laptop is best?'.",
        # --- Technical: Code & Programming ---
        "programming language syntax, data types like string or integer, algorithm logic, function, method, programming class, object-oriented paradigm, variable scope, control flow, import, module, package, library, framework, recursion, iteration",
        "software design patterns, creational: singleton, factory, builder; structural: adapter, decorator, facade, proxy; behavioral: observer, strategy, command, mediator, chain of responsibility; abstract interface, polymorphism, composition",
        "error handling, exception, stack trace, TypeError, NullPointerException, IndexError, segmentation fault, core dump, stack overflow, runtime vs compile-time error, assertion failed, syntax error, null pointer dereference, memory leak, bug",
        "HTTP status codes: 404 Not Found, 500 Internal Server Error, 403 Forbidden, 401 Unauthorized, 200 OK, 201 Created. API response, 502 Bad Gateway, 503 Service Unavailable, 400 Bad Request, 429 Too Many Requests, timeout, CORS",
        "API design, endpoint, REST, GraphQL, SOAP, RPC. HTTP methods: GET, POST, PUT, DELETE, PATCH. Request-response cycle, payload, authentication token, bearer, JWT, OAuth, API key, query parameters, path variables, request body",
        # --- Technical: DevOps, CLI & Data ---
        "terminal command line shell prompt, bash, zsh, powershell, cmd. Filesystem navigation: cd, ls, pwd. File management: mkdir, rm, cp, mv, chmod, chown. Text processing with grep, sed, awk, cat. User permissions: sudo, root access",
        "developer CLI tools, package manager, install, update. Network requests with curl, wget. Secure shell access with SSH. Version control with git: clone, commit, push, pull. Containerization with docker: run, build, compose; npm, pip",
        "data interchange formats, serialization, deserialization, parsing. JSON object, array, key-value pair. XML tags, attributes. YAML indentation, TOML, CSV, .ini properties. Config file, env variables, dictionary, map, protocol buffers",
        "WebSocket real-time bidirectional communication, server-client connection on a port, binary message protocol, handshake, HTTP upgrade, socket programming, TCP, UDP, listening, binding, accepting, streaming, pub-sub, broadcast channel",
        "file system path, directory structure, config log bin, absolute vs relative path, operating system, filesystem, mount point, home, /tmp, /var, shared library, symbolic link, inode, file permissions, owner, group, read write execute",
        "container orchestration, cluster management, service scaling, replication, load balancing, namespace, pod, deployment, infrastructure, Kubernetes (K8s), Docker Swarm, container runtime (CRI-O, containerd), image registry, Dockerfile",
        "querying a database, SQL statement, database table, column, row, index, primary key, foreign key relationship, join, filter, select, insert, update, delete, relational vs NoSQL, MongoDB, PostgreSQL, MySQL, Redis, schema, transaction",
        "application logging, log output, stack trace levels like INFO, WARN, ERROR, DEBUG, FATAL. Log message components: timestamp, module, line number. Diagnostic telemetry, monitoring, and observability for system health and debugging",
        # --- Technical: Algorithms & Testing ---
        "algorithm analysis, O(log n) time complexity, space complexity, data structures, hash table, array, linked list, queue, stack, heap, priority queue, graph, adjacency matrix, depth-first search (DFS), breadth-first search (BFS)",
        "sorting algorithms performance and implementation, including merge sort, quicksort, insertion sort, selection sort. Understanding stable vs unstable sorts, in-place operations, comparison-based sorting, and computational complexity",
        "regex pattern, regular expression matching, groups, capturing, backslash escapes, metacharacters, wildcards, quantifiers, character classes, lookaheads, lookbehinds, alternation, anchors, word boundary, multiline flag, global search",
        "software testing, unit test, assertion, mock, stub, fixture, test suite, test case, verification, automated QA, validation framework, JUnit, pytest, Jest. Integration, end-to-end (E2E), functional, regression, acceptance testing",
        "cloud computing platforms, infrastructure as a service (IaaS), PaaS, AWS, Azure, GCP, compute instance, region, availability zone, elasticity, distributed system, virtual machine, container, serverless, Lambda, edge computing, CDN",
        "markdown syntax for text formatting, horizontal rule, separator using dashes, headings, fenced code block with triple backticks, inline code, emphasis with bold and italic, strikethrough, blockquote, nested list, task list, markdown table",
        "code formatting and style, indentation with whitespace, tabs vs spaces, nested function body, class method, structured code, syntax highlighting for languages like Python, JavaScript, Java, C++, Go, Rust, TypeScript, Prettier, ESLint",
        # --- Instructional: Formatting & Rewriting ---
        "format the output as structured data. Return the answer as JSON with specific keys and values, or as YAML. Organize information into a CSV file or a database-style table with columns and rows. Present as a list of objects or an array.",
        "style the text presentation. Use markdown formatting like bullet points, a numbered list, or a task list. Organize content into a grid or tabular layout with proper alignment. Create a hierarchical structure with nested elements for clarity.",
        "adjust the response length. Make the answer shorter, more concise, brief, or condensed. Summarize the key points. Trim down the text to reduce the overall word count or meet a specific character limit. Be less verbose and more direct.",
        "change the explanation depth. Make the response more detailed, comprehensive, and elaborate. Expand on previous points and go into more depth. Provide a thorough, in-depth analysis. Explain the topic with more complexity and nuance.",
        "rewrite the previous response. Rephrase, paraphrase, or reformulate the answer using different wording. Restate the information in another way to offer an alternative perspective. Express the same meaning but with a new structure or vocabulary.",
        "alter the response tone. Change the writing style to be more formal, academic, or professional. Alternatively, make it more casual, friendly, and conversational. Adapt the register and voice to suit a specific audience or context level.",
        "explain the concept in simpler terms. Break down the topic step-by-step for a beginner. Clarify a confusing point. Explain it like I'm five years old (ELI5). Use an analogy or a concrete example to help me understand the idea clearly.",
        "continue the generated response. Keep going with the explanation or list. Provide more information and finish your thought. Complete the rest of the content or story. Proceed with the next steps. Do not stop until you have concluded.",
        "act as a specific persona or role. Respond as if you were a pirate, a scientist, or a travel guide. Adopt the character's voice, style, and knowledge base in your answer. Maintain the persona throughout the entire response.",
        "compare and contrast two or more topics. Explain the similarities and differences between A and B. Provide a detailed analysis of what they have in common and how they diverge. Create a table to highlight the key distinctions.",
        # --- Instructional: Math & Calculation ---
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
        # --- Instructional: Translation ---
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
        # --- Instructional: Proofreading & Editing ---
        "proofread, review, revise, or edit provided text for errors. Here is my draft, check it for typos and mistakes. Correct grammar, spelling, punctuation. Review emails, essays, documents, reviews, or any submitted content for clarity and flow.",
        "proofread for coherence, readability, or professionalism. Polish the text to ensure it sounds professional and is free of errors. Check for textual quality, sentence structure, and overall writing effectiveness in submitted drafts or passages.",
        "correct grammatical issues like subject-verb agreement, incorrect verb tense, pronoun reference errors, misplaced modifiers, or faulty sentence structure. Validate if a sentence is grammatically correct. Check word usage (their/there/they're).",
        "fix passive voice, run-on sentences, comma splices, or sentence fragments. Address punctuation errors with apostrophes, quotation marks, periods, semicolons, dashes. Ensure proper capitalization and resolve structural writing problems.",
        "improve writing quality: suggest better word choice, alternative phrasing, synonyms, or refined expression. Enhance vocabulary and diction. Make writing more direct, concise, engaging, smooth, or readable. Restructure sentences for coherence.",
        "remove wordiness, filler words, or redundancy from text. Improve logical progression of ideas. Eliminate awkward phrasing. Make the writing flow better and ensure ideas connect seamlessly for better overall quality and readability.",
        "rewrite, rephrase, paraphrase, or reformulate text using different wording. Restate information in another way. Express the same meaning but with new structure or vocabulary. Adapt tone to be more formal, academic, or professional.",
        "adapt writing tone to be more casual, friendly, or conversational. Change the register and voice to suit a specific audience or context level. Adjust the writing style while maintaining the core message and information presented in the original text.",
        # --- Transient States & Momentary Situations ---
        "describing current temporary emotional states, fleeting feelings, or momentary moods without lasting significance. 'I'm feeling stressed this week', 'I'm tired today', 'I'm excited right now', 'I'm frustrated at the moment', 'I'm happy'.",
        "temporary emotions or passing states that are not enduring personal facts. 'I'm angry about this situation', 'I'm nervous about tomorrow', 'I feel great today', 'I'm worried right now'. These are transient feelings, not biographical information.",
        "mentioning one-time events, temporary situations, or transient circumstances without lasting impact. 'I have a presentation on Friday', 'I'm at the store', 'I'm working late tonight', 'I ate pizza for lunch', 'It's raining here today'.",
        "describing momentary situations, current locations, or immediate activities. 'I'm in a meeting', 'I'm driving to work', 'I'm cooking dinner', 'I'm watching a movie'. These are temporary circumstances, not biographical or lasting personal facts.",
    ]

    PERSONAL_CATEGORY_DESCRIPTIONS = [
        "**Identity Core:** Directly stating facts about my name, birthdate, age, nationality, ethnicity, personality traits, core beliefs, values, religion, cultural background, education history, academic degrees, or formative personal experiences.",
        "**Medical History:** Directly stating facts about my medical diagnoses, chronic conditions, past surgeries or medical procedures, medications I currently take or have taken in the past, supplements I use, vision or hearing conditions, allergies.",
        "**Physical Health:** Directly stating facts about my dietary restrictions, physical measurements like height and weight, fitness routines, sleep patterns, physical appearance, mental health conditions, ongoing symptoms, or wellness practices I follow.",
        "**Family & Relationships:** Directly stating facts about my family members including their names, ages, relationships to me, occupations, or health conditions. Information about my spouse, partner, children, friends, or romantic relationship status.",
        "**Social & Pets:** Directly stating facts about my pets including their names and breeds, social activities I participate in, community involvement, or details about the people in my life and how I interact with my social circle and broader community.",
        "**Job & Workplace:** Directly stating facts about my current or past job titles, employer names, workplace, industry, career transitions, professional certifications, technical skills I possess, colleagues, or work arrangements like remote or hybrid.",
        "**Professional Growth:** Directly stating facts about professional development activities I'm engaged in or completed, training programs, career milestones, work history, or any facts related to my professional life, expertise, and career trajectory.",
        "**Finance & Legal:** Directly stating facts about my financial situation, income level, budgeting constraints, investments I hold, savings goals, debts I have, tax situations, legal matters I'm involved in, or financial obligations and commitments.",
        "**Home & Location:** Directly stating facts about my residence type, living arrangements, roommates, neighborhood, city, country, relocations I've made, commute details, vehicles I own or drive with make/model/year, or transportation methods I use.",
        "**Hobbies & Activities:** Directly stating facts about my hobbies, recreational activities, creative projects I work on, sports I play or follow, specific media preferences like favorite movies, books, music genres, games, or personal collections.",
        "**Leisure & Entertainment:** Directly stating facts about pastimes I regularly engage in, entertainment preferences, artistic pursuits, leisure activities, or any facts about how I spend my free time and what I enjoy doing for relaxation or fulfillment.",
        "**Future Plans:** Directly stating facts about my scheduled future plans, confirmed appointments, upcoming events I'm attending, booked travel itineraries, stated long-term personal goals, career aspirations, or life milestones I'm working toward.",
        "**Life Events:** Directly stating facts about my past life events, significant personal milestones like graduations, marriages, or births, historical medical events, previous jobs or living situations, travel history, or memorable experiences.",
        "**Personal History:** Directly stating facts about memorable personal experiences with temporal context, past achievements, formative moments, historical facts about my life journey, or biographical information about my past that shaped who I am.",
        "**Emotional Landscape:** Directly stating my current emotional state toward lasting situations, not momentary feelings. My attitudes toward specific people or relationships, deep-seated preferences, strong aversions or dislikes, or motivations.",
        "**Inner Life:** Directly stating facts about sources of stress or joy in my life, ongoing emotional experiences, persistent feelings about important matters, or enduring attitudes and perspectives that reflect my emotional landscape and inner life.",
        "**Possessions & Brands:** Directly stating facts about specific items I own like devices, appliances, or vehicles with details. Products I regularly use or consume, brands I prefer, subscriptions I have, or material possessions with identifying details.",
    ]

    class SkipReason(Enum):
        SKIP_SIZE = "SKIP_SIZE"
        SKIP_NON_PERSONAL = "SKIP_NON_PERSONAL"

    STATUS_MESSAGES = {
        SkipReason.SKIP_SIZE: "📏 Message Length Out of Limits, skipping memory operations",
        SkipReason.SKIP_NON_PERSONAL: "🚫 Non-Personal Content Detected, skipping memory operations",
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

        # Pattern 1: Multiple URLs (5+ full URLs indicates link lists or technical references)
        url_pattern_count = message.count("http://") + message.count("https://")
        if url_pattern_count >= 5:
            return True

        # Pattern 2: Long unbroken alphanumeric strings (tokens, hashes, base64)
        words = message.split()
        for word in words:
            cleaned = word.strip('.,;:!?()[]{}"\'"')
            if len(cleaned) > 80 and cleaned.replace("-", "").replace("_", "").isalnum():
                return True

        # Pattern 3: Markdown/text separators (repeated ---, ===, ___, ***)
        separator_patterns = ["---", "===", "___", "***"]
        for pattern in separator_patterns:
            if message.count(pattern) >= 2:
                return True

        # Pattern 4: Command-line patterns with context-aware detection
        lines_stripped = [line.strip() for line in message.split("\n") if line.strip()]
        if lines_stripped:
            actual_command_lines = 0
            for line in lines_stripped:
                if line.startswith("$ ") and len(line) > 2:
                    parts = line[2:].split()
                    if parts and parts[0].isalnum():
                        actual_command_lines += 1
                elif "$ " in line:
                    dollar_index = line.find("$ ")
                    if dollar_index > 0 and line[dollar_index - 1] in (" ", ":", "\t"):
                        parts = line[dollar_index + 2 :].split()
                        if parts and len(parts[0]) > 0 and (parts[0].isalnum() or parts[0] in ["curl", "wget", "git", "npm", "pip", "docker"]):
                            actual_command_lines += 1
                elif line.startswith("# ") and len(line) > 2:
                    rest = line[2:].strip()
                    if rest and not rest[0].isupper() and " " in rest:
                        actual_command_lines += 1
                elif line.startswith("> ") and len(line) > 2:
                    pass

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
        line_count = message.count("\n")
        if line_count >= 8:
            lines = message.split("\n")
            non_empty_lines = [line for line in lines if line.strip()]
            if non_empty_lines:
                colon_lines = sum(1 for line in non_empty_lines if ":" in line and not line.strip().startswith("#"))
                indented_lines = sum(1 for line in non_empty_lines if line.startswith((" ", "\t")))

                if colon_lines / len(non_empty_lines) > 0.4 and indented_lines / len(non_empty_lines) > 0.5:
                    words_outside_kv = 0
                    for line in non_empty_lines:
                        if ":" not in line:
                            words_outside_kv += len(line.split())

                    if words_outside_kv < 5:
                        return True

        # Pattern 8: Highly structured multi-line content (require markup chars for technical confidence)
        if line_count > 15:
            lines = message.split("\n")
            non_empty_lines = [line for line in lines if line.strip()]
            if non_empty_lines:
                markup_in_lines = sum(1 for line in non_empty_lines if any(c in line for c in "{}[]<>"))
                structured_lines = sum(1 for line in non_empty_lines if line.startswith((" ", "\t")))

                if markup_in_lines / len(non_empty_lines) > 0.3:
                    return True
                elif structured_lines / len(non_empty_lines) > 0.6:
                    operators = ["=", "+", "-", "*", "/", "<", ">", "&", "|", "!", ":", "?"]
                    operator_count = sum(message.count(op) for op in operators)
                    if (operator_count / msg_len) > 0.05:
                        return True

        # Pattern 9: Code-like indentation pattern (require code indicators to avoid false positives from bullet lists)
        if line_count >= 3:
            lines = message.split("\n")
            non_empty_lines = [line for line in lines if line.strip()]
            if non_empty_lines:
                indented_lines = sum(1 for line in non_empty_lines if line[0] in (" ", "\t"))
                if indented_lines / len(non_empty_lines) > 0.5:
                    code_ending_chars = ["{", "}", "(", ")", ";"]
                    lines_with_code_endings = sum(1 for line in non_empty_lines if line.strip().endswith(tuple(code_ending_chars)))
                    if lines_with_code_endings / len(non_empty_lines) > 0.2:
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

    async def detect_skip_reason(self, message: str, max_message_chars: int, memory_system: "Filter") -> Optional[str]:
        """
        Detect if a message should be skipped using two-stage detection:
        1. Fast-path structural patterns (~95% confidence)
        2. Binary semantic classification (personal vs non-personal)
        Returns:
            Skip reason string if content should be skipped, None otherwise
        """
        size_issue = self.validate_message_size(message, max_message_chars)
        if size_issue:
            return size_issue

        fast_skip = self._fast_path_skip_detection(message)
        if fast_skip:
            logger.info(f"⚡ Fast-path skip: {self.SkipReason.SKIP_NON_PERSONAL.value}")
            return self.SkipReason.SKIP_NON_PERSONAL.value

        if self._reference_embeddings is None:
            await self.initialize()

        message_embedding_result = await self.embedding_function([message.strip()])
        message_embedding = np.array(message_embedding_result[0])

        personal_similarities = np.dot(message_embedding, self._reference_embeddings["personal"].T)
        max_personal_similarity = float(personal_similarities.max())

        non_personal_similarities = np.dot(message_embedding, self._reference_embeddings["non_personal"].T)
        max_non_personal_similarity = float(non_personal_similarities.max())

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
    ) -> List[Dict]:
        """Use LLM to select most relevant memories."""
        memory_lines = self.memory_system._format_memories_for_llm(candidate_memories)
        memory_context = "\n".join(memory_lines)

        user_prompt = f"""CURRENT DATE/TIME: {self.memory_system.format_current_datetime()}

USER MESSAGE: {user_message}

CANDIDATE MEMORIES:
{memory_context}"""

        try:
            response = await self.memory_system._query_llm(
                Prompts.MEMORY_RERANKING,
                user_prompt,
                response_model=Models.MemoryRerankingResponse,
            )

            selected_memories = []
            for memory in candidate_memories:
                if memory["id"] in response.ids and len(selected_memories) < max_count:
                    selected_memories.append(memory)

            logger.info(f"🧠 LLM selected {len(selected_memories)} out of {len(candidate_memories)} candidates")

            return selected_memories

        except Exception as e:
            logger.warning(f"🤖 LLM reranking failed during memory relevance analysis: {str(e)}")
            return candidate_memories

    async def rerank_memories(
        self,
        user_message: str,
        candidate_memories: List[Dict],
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
                f"🤖 LLM Analyzing {len(llm_candidates)} Memories for Relevance",
                done=False,
                level=Constants.STATUS_LEVEL["Intermediate"],
            )
            logger.info(f"🧠 Using LLM reranking: {decision_reason}")

            selected_memories = await self._llm_select_memories(user_message, llm_candidates, max_injection)

            if not selected_memories:
                logger.info("📭 No relevant memories after LLM analysis")
                await self.memory_system._emit_status(
                    emitter, f"📭 No Relevant Memories After LLM Analysis", done=True, level=Constants.STATUS_LEVEL["Intermediate"]
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

    async def _check_semantic_duplicate(self, content: str, existing_memories: List, user_id: str) -> Optional[str]:
        """
        Check if content is semantically duplicate of existing memories using embeddings.
        Returns the ID of duplicate memory if found, None otherwise.
        """
        valid_memories = [m for m in existing_memories if m.content and len(m.content.strip()) >= Constants.MIN_MESSAGE_CHARS]
        if not valid_memories:
            return None

        memory_contents = [m.content for m in valid_memories]
        all_texts = [content] + memory_contents
        all_embeddings = await self.memory_system._generate_embeddings(all_texts, user_id)

        content_embedding = all_embeddings[0]
        for i, memory in enumerate(valid_memories):
            memory_embedding = all_embeddings[i + 1]
            if memory_embedding is None:
                continue
            similarity = float(np.dot(content_embedding, memory_embedding))
            if similarity >= Constants.DEDUPLICATION_SIMILARITY_THRESHOLD:
                logger.info(f"🔍 Semantic duplicate detected: similarity={similarity:.3f} with memory {memory.id}")
                return str(memory.id)

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
        except Exception as e:
            logger.error(f"💾 Failed to retrieve user memories: {str(e)}")
            return []

        if not user_memories:
            logger.info("💭 No existing memories found for consolidation")
            return []

        logger.info(f"🚀 Processing {len(user_memories)} cached memories for consolidation")

        try:
            all_similarities, _, _ = await self.memory_system._compute_similarities(user_message, user_id, user_memories)
        except Exception as e:
            logger.error(f"🔍 Failed to compute memory similarities: {str(e)}")
            return []

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
        emitter: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
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
                self.memory_system._query_llm(
                    Prompts.MEMORY_CONSOLIDATION,
                    user_prompt,
                    response_model=Models.ConsolidationResponse,
                ),
                timeout=Constants.LLM_CONSOLIDATION_TIMEOUT_SEC,
            )
        except Exception as e:
            logger.warning(f"🤖 LLM consolidation failed during memory processing: {str(e)}")
            await self.memory_system._emit_status(emitter, f"⚠️ Memory Consolidation Failed", done=True, level=Constants.STATUS_LEVEL["Basic"])
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

        deduplicated_operations = []
        seen_contents = set()
        seen_update_ids = set()

        for op in operations:
            if not op.validate_operation(existing_memory_ids):
                continue

            if op.operation == Models.MemoryOperationType.UPDATE and op.id in seen_update_ids:
                logger.info(f"⏭️ Skipping duplicate UPDATE for memory {op.id}")
                continue

            if op.operation in [Models.MemoryOperationType.CREATE, Models.MemoryOperationType.UPDATE]:
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
        self, operations: List, current_memories: List, user_id: str, operation_type: str, delete_operations: Optional[List] = None
    ) -> List:
        """
        Deduplicate operations against existing memories using semantic similarity.
        For UPDATE operations, preserves enriched content and deletes the duplicate.
        """
        deduplicated = []

        for operation in operations:
            memories_to_check = current_memories
            if operation_type == "UPDATE":
                memories_to_check = [m for m in current_memories if str(m.id) != operation.id]

            duplicate_id = await self._check_semantic_duplicate(operation.content, memories_to_check, user_id)

            if duplicate_id:
                if operation_type == "UPDATE" and delete_operations is not None:
                    logger.info(f"🔄 UPDATE creates duplicate: keeping {operation.id}, deleting {duplicate_id}")
                    deduplicated.append(operation)
                    delete_operations.append(Models.MemoryOperation(operation=Models.MemoryOperationType.DELETE, content="", id=duplicate_id))
                else:
                    logger.info(f"⏭️ Skipping duplicate {operation_type}: {self.memory_system._truncate_content(operation.content)} (matches {duplicate_id})")
                continue

            deduplicated.append(operation)

        return deduplicated

    async def execute_memory_operations(self, operations: List[Dict[str, Any]], user_id: str, emitter: Optional[Callable] = None) -> Tuple[int, int, int, int]:
        """Execute consolidation operations with simplified tracking."""
        if not operations:
            return 0, 0, 0, 0

        user = await asyncio.wait_for(
            asyncio.to_thread(Users.get_user_by_id, user_id),
            timeout=Constants.DATABASE_OPERATION_TIMEOUT_SEC,
        )

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

        user_memories = await self.memory_system._get_user_memories(user_id)
        memory_contents_for_deletion = {str(mem.id): mem.content for mem in user_memories} if operations_by_type["DELETE"] else {}

        if operations_by_type["CREATE"]:
            operations_by_type["CREATE"] = await self._deduplicate_operations(operations_by_type["CREATE"], user_memories, user_id, operation_type="CREATE")

        if operations_by_type["UPDATE"]:
            operations_by_type["UPDATE"] = await self._deduplicate_operations(
                operations_by_type["UPDATE"], user_memories, user_id, operation_type="UPDATE", delete_operations=operations_by_type["DELETE"]
            )

        for operation_type, ops in operations_by_type.items():
            if not ops:
                continue

            batch_tasks = []
            for operation in ops:
                task = self.memory_system._execute_single_operation(operation, user)
                batch_tasks.append(task)

            results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for idx, result in enumerate(results):
                operation = ops[idx]

                if isinstance(result, Exception):
                    failed_count += 1
                    await self.memory_system._emit_status(emitter, f"❌ Failed {operation_type}", done=False, level=Constants.STATUS_LEVEL["Intermediate"])
                elif result == Models.MemoryOperationType.CREATE.value:
                    created_count += 1
                    content_preview = self.memory_system._truncate_content(operation.content)
                    await self.memory_system._emit_status(emitter, f"📝 Created: {content_preview}", done=False, level=Constants.STATUS_LEVEL["Intermediate"])
                elif result == Models.MemoryOperationType.UPDATE.value:
                    updated_count += 1
                    content_preview = self.memory_system._truncate_content(operation.content)
                    await self.memory_system._emit_status(emitter, f"✏️ Updated: {content_preview}", done=False, level=Constants.STATUS_LEVEL["Intermediate"])
                elif result == Models.MemoryOperationType.DELETE.value:
                    deleted_count += 1
                    content_preview = memory_contents_for_deletion.get(operation.id, operation.id)
                    if content_preview and content_preview != operation.id:
                        content_preview = self.memory_system._truncate_content(content_preview)
                    await self.memory_system._emit_status(emitter, f"🗑️ Deleted: {content_preview}", done=False, level=Constants.STATUS_LEVEL["Intermediate"])
                elif result in [
                    Models.OperationResult.FAILED.value,
                    Models.OperationResult.UNSUPPORTED.value,
                ]:
                    failed_count += 1
                    await self.memory_system._emit_status(emitter, f"❌ Failed {operation_type}", done=False, level=Constants.STATUS_LEVEL["Intermediate"])

        total_executed = created_count + updated_count + deleted_count
        logger.info(
            f"✅ Memory processing completed: {total_executed}/{len(operations)} ops (created: {created_count}, updated: {updated_count}, deleted: {deleted_count}, failed: {failed_count})"
        )

        if total_executed > 0:
            operation_details = self.memory_system._build_operation_details(created_count, updated_count, deleted_count)
            logger.info(f"🔄 Memory operations: {', '.join(operation_details)}")
            await self.memory_system._refresh_user_cache(user_id)

        return created_count, updated_count, deleted_count, failed_count

    async def run_consolidation_pipeline(
        self,
        user_message: str,
        user_id: str,
        emitter: Optional[Callable] = None,
        cached_similarities: Optional[List[Dict[str, Any]]] = None,
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

                    await self.memory_system._emit_status(emitter, operations_summary, done=True, level=Constants.STATUS_LEVEL["Basic"])
            else:
                duration = time.time() - start_time
                await self.memory_system._emit_status(
                    emitter,
                    f"✅ Consolidation Complete: No Updates Needed",
                    done=True,
                    level=Constants.STATUS_LEVEL["Detailed"],
                )

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

    async def _set_pipeline_context(
        self,
        __event_emitter__: Optional[Callable] = None,
        __user__: Optional[Dict[str, Any]] = None,
        __model__: Optional[str] = None,
        __request__: Optional[Request] = None,
    ) -> None:
        """Set pipeline context parameters to avoid duplication in inlet/outlet methods."""
        if __event_emitter__:
            self.__current_event_emitter__ = __event_emitter__
        if __user__:
            self.__user__ = __user__
        if __model__:
            self.__model__ = __model__
            if self.valves.memory_model:
                logger.info(f"🤖 Using custom memory model: {__model__}")
        if __request__:
            self.__request__ = __request__

            if self._embedding_function is None and hasattr(__request__.app.state, "EMBEDDING_FUNCTION"):
                self._embedding_function = __request__.app.state.EMBEDDING_FUNCTION
                logger.info("✅ Using OpenWebUI embedding function")

            if self._embedding_function and self._embedding_dimension is None:
                async with self._initialization_lock:
                    if self._embedding_dimension is None:
                        await self._detect_embedding_dimension()

            if self._embedding_function and self._skip_detector is None:
                global _SHARED_SKIP_DETECTOR_CACHE, _SHARED_SKIP_DETECTOR_CACHE_LOCK
                embedding_engine = getattr(__request__.app.state.config, "RAG_EMBEDDING_ENGINE", "")
                embedding_model = getattr(__request__.app.state.config, "RAG_EMBEDDING_MODEL", "")
                cache_key = f"{embedding_engine}:{embedding_model}"

                async with _SHARED_SKIP_DETECTOR_CACHE_LOCK:
                    if cache_key in _SHARED_SKIP_DETECTOR_CACHE:
                        logger.info(f"♻️ Reusing cached skip detector: {cache_key}")
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

    def _get_last_user_message(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Extract the last user message text from a list of messages."""
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content", "")
            text = self._extract_text_from_content(content)
            if text:
                return text
        return None

    def _validate_system_configuration(self) -> None:
        """Validate configuration and fail if invalid."""
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
        return hashlib.sha256(str(text).encode()).hexdigest()

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
        return embedding / norm if norm > 0 else embedding

    async def _generate_embeddings(self, texts: Union[str, List[str]], user_id: str) -> Union[np.ndarray, List[np.ndarray]]:
        """Unified embedding generation for single text or batch with optimized caching using OpenWebUI's embedding function."""
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts

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
            user = await asyncio.to_thread(Users.get_user_by_id, user_id) if hasattr(self, "__user__") else None

            raw_embeddings = await self._embedding_function(uncached_texts, prefix=None, user=user)

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

    async def _should_skip_memory_operations(self, user_message: str) -> Tuple[bool, str]:
        skip_reason = await self._skip_detector.detect_skip_reason(user_message, Constants.MAX_MESSAGE_CHARS, memory_system=self)
        if skip_reason:
            status_key = SkipDetector.SkipReason(skip_reason)
            return True, SkipDetector.STATUS_MESSAGES[status_key]
        return False, ""

    async def _process_user_message(self, body: Dict[str, Any]) -> Tuple[Optional[str], bool, str]:
        """Extract user message and determine if memory operations should be skipped."""
        user_message = self._get_last_user_message(body["messages"])
        if not user_message:
            return None, True, SkipDetector.STATUS_MESSAGES[SkipDetector.SkipReason.SKIP_SIZE]

        should_skip, skip_reason = await self._should_skip_memory_operations(user_message)
        return user_message, should_skip, skip_reason

    async def _get_user_memories(self, user_id: str) -> List:
        """Get user memories with timeout handling."""
        memories = await asyncio.wait_for(
            asyncio.to_thread(Memories.get_memories_by_user_id, user_id),
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
        median_score = float(np.median(scores))

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
            content_hash = hashlib.sha256(str(content).encode("utf-8")).hexdigest()[: Constants.CACHE_KEY_HASH_PREFIX_LENGTH]
            return f"{cache_type}_{user_id}:{content_hash}"
        return f"{cache_type}_{user_id}"

    def format_current_datetime(self) -> str:
        """Return current UTC datetime in human-readable format."""
        return datetime.now(timezone.utc).strftime("%A %B %d %Y at %H:%M:%S UTC")

    def _format_memories_for_llm(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Format memories for LLM consumption with hybrid format and human-readable timestamps."""
        memory_lines = []
        for memory in memories:
            line = f"[{memory['id']}] {memory['content']}"
            record_date = memory.get("updated_at") or memory.get("created_at")
            if record_date:
                try:
                    if isinstance(record_date, str):
                        parsed_date = datetime.fromisoformat(record_date.replace("Z", "+00:00"))
                    else:
                        parsed_date = record_date
                    formatted_date = parsed_date.strftime("%b %d %Y")
                    line += f" [noted at {formatted_date}]"
                except Exception as e:
                    logger.warning(f"📅 Failed to format date {record_date}: {str(e)}")
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
        user_memories: Optional[List] = None,
        emitter: Optional[Callable] = None,
        cached_similarities: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Retrieve memories for injection using similarity computation with optional LLM reranking."""
        if cached_similarities is not None:
            memories = [m for m in cached_similarities if m.get("relevance", 0) >= self.valves.semantic_retrieval_threshold]
            logger.info(f"🔍 Using cached similarities: {len(memories)} candidates")
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
            await self._emit_status(emitter, "📭 No Memories Found", done=True, level=Constants.STATUS_LEVEL["Intermediate"])
            return {"memories": [], "threshold": None}

        memories, threshold, all_similarities = await self._compute_similarities(user_message, user_id, user_memories)

        if memories:
            final_memories, reranking_info = await self._llm_reranking_service.rerank_memories(user_message, memories, emitter)
        else:
            logger.info("📭 No relevant memories found above similarity threshold")
            await self._emit_status(emitter, "📭 No Relevant Memories Found", done=True, level=Constants.STATUS_LEVEL["Intermediate"])
            final_memories = memories
            reranking_info = {"llm_decision": False, "decision_reason": "no_candidates"}

        self._log_retrieved_memories(final_memories, "semantic")

        return {
            "memories": final_memories,
            "threshold": threshold,
            "all_similarities": all_similarities,
            "reranking_info": reranking_info,
        }

    async def _add_memory_context(
        self,
        body: Dict[str, Any],
        memories: Optional[List[Dict[str, Any]]] = None,
        user_id: Optional[str] = None,
        emitter: Optional[Callable] = None,
    ) -> None:
        """Add memory context to request body with simplified logic."""
        content_parts = [f"Current Date/Time: {self.format_current_datetime()}"]

        memory_count = 0
        if memories and user_id:
            memory_count = len(memories)
            memory_header = f"CONTEXT: The following {'fact' if memory_count == 1 else 'facts'} about the user are provided for background only. Not all facts may be relevant to the current request."
            formatted_memories = []

            for idx, memory in enumerate(memories, 1):
                formatted_memory = f"- {' '.join(memory['content'].split())}"
                formatted_memories.append(formatted_memory)

                content_preview = self._truncate_content(memory["content"])
                await self._emit_status(emitter, f"💭 {idx}/{memory_count}: {content_preview}", done=False, level=Constants.STATUS_LEVEL["Intermediate"])

            memory_footer = "IMPORTANT: Do not mention or imply you received this list. These facts are for background context only."
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
            "id": str(memory.id),
            "content": memory.content,
            "relevance": similarity,
        }
        if hasattr(memory, "created_at") and memory.created_at:
            memory_dict["created_at"] = datetime.fromtimestamp(memory.created_at, tz=timezone.utc).isoformat()
        if hasattr(memory, "updated_at") and memory.updated_at:
            memory_dict["updated_at"] = datetime.fromtimestamp(memory.updated_at, tz=timezone.utc).isoformat()
        return memory_dict

    async def _compute_similarities(self, user_message: str, user_id: str, user_memories: List) -> Tuple[List[Dict], float, List[Dict]]:
        """Compute similarity scores between user message and memories."""
        if not user_memories:
            return [], self.valves.semantic_retrieval_threshold, []

        query_embedding = await self._generate_embeddings(user_message, user_id)
        memory_contents = [memory.content for memory in user_memories]
        memory_embeddings = await self._generate_embeddings(memory_contents, user_id)

        memory_data = []
        for memory_index, memory in enumerate(user_memories):
            memory_embedding = memory_embeddings[memory_index]
            if memory_embedding is None:
                continue

            similarity = float(np.dot(query_embedding, memory_embedding))
            memory_dict = self._build_memory_dict(memory, similarity)
            memory_data.append(memory_dict)

        memory_data.sort(key=lambda x: x["relevance"], reverse=True)

        threshold = self.valves.semantic_retrieval_threshold
        filtered_memories = [m for m in memory_data if m["relevance"] >= threshold]
        return filtered_memories, threshold, memory_data

    async def inlet(
        self,
        body: Dict[str, Any],
        __event_emitter__: Optional[Callable] = None,
        __user__: Optional[Dict[str, Any]] = None,
        __request__: Optional[Request] = None,
    ) -> Dict[str, Any]:
        """Simplified inlet processing for memory retrieval and injection."""

        model_to_use = self.valves.memory_model or (body.get("model") if isinstance(body, dict) else None)
        await self._set_pipeline_context(__event_emitter__, __user__, model_to_use, __request__)

        user_id = __user__.get("id") if body and __user__ else None
        if not user_id:
            return body

        user_message, should_skip, skip_reason = await self._process_user_message(body)

        skip_cache_key = self._cache_key(self._cache_manager.SKIP_STATE_CACHE, user_id, user_message)
        await self._cache_manager.put(
            user_id,
            self._cache_manager.SKIP_STATE_CACHE,
            skip_cache_key,
            should_skip,
        )

        if not user_message or should_skip:
            if __event_emitter__ and skip_reason:
                await self._emit_status(__event_emitter__, skip_reason, done=True, level=Constants.STATUS_LEVEL["Intermediate"])
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
            retrieval_result = await self._retrieve_relevant_memories(user_message, user_id, user_memories, __event_emitter__)
            memories = retrieval_result.get("memories", [])
            threshold = retrieval_result.get("threshold")
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

        model_to_use = self.valves.memory_model or (body.get("model") if isinstance(body, dict) else None)
        await self._set_pipeline_context(__event_emitter__, __user__, model_to_use, __request__)

        user_id = __user__.get("id") if body and __user__ else None
        if not user_id:
            return body

        user_message = self._get_last_user_message(body.get("messages", []))
        if not user_message:
            return body

        skip_cache_key = self._cache_key(self._cache_manager.SKIP_STATE_CACHE, user_id, user_message)
        should_skip = await self._cache_manager.get(user_id, self._cache_manager.SKIP_STATE_CACHE, skip_cache_key)

        if should_skip:
            logger.info("⏭️ Skipping outlet: inlet detected skip condition")
            return body

        retrieval_cache_key = self._cache_key(self._cache_manager.RETRIEVAL_CACHE, user_id, user_message)
        cached_similarities = await self._cache_manager.get(user_id, self._cache_manager.RETRIEVAL_CACHE, retrieval_cache_key)
        task = asyncio.create_task(self._llm_consolidation_service.run_consolidation_pipeline(user_message, user_id, __event_emitter__, cached_similarities))
        self._background_tasks.add(task)

        def safe_cleanup(t: asyncio.Task) -> None:
            try:
                self._background_tasks.discard(t)
                if t.exception() and not t.cancelled():
                    exception = t.exception()
                    logger.error(f"❌ Background consolidation failed: {str(exception)}")
            except Exception as e:
                logger.error(f"❌ Failed to cleanup background task: {str(e)}")

        task.add_done_callback(safe_cleanup)
        return body

    async def shutdown(self) -> None:
        """Cleanup method to properly shutdown background tasks."""
        self._shutdown_event.set()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

        await self._cache_manager.clear_all_caches()

    async def _refresh_user_cache(self, user_id: str) -> None:
        """Refresh user cache - clear stale caches and update with fresh embeddings."""
        start_time = time.time()
        try:
            retrieval_cleared = await self._cache_manager.clear_user_cache(user_id, self._cache_manager.RETRIEVAL_CACHE)
            embedding_cleared = await self._cache_manager.clear_user_cache(user_id, self._cache_manager.EMBEDDING_CACHE)
            skip_state_cleared = await self._cache_manager.clear_user_cache(user_id, self._cache_manager.SKIP_STATE_CACHE)
            logger.info(f"🔄 Cleared cache: {retrieval_cleared} retrieval, {embedding_cleared} embedding, {skip_state_cleared} skip entries")

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

            memory_contents = [memory.content for memory in user_memories if len(memory.content.strip()) >= Constants.MIN_MESSAGE_CHARS]

            if memory_contents:
                await self._generate_embeddings(memory_contents, user_id)
                duration = time.time() - start_time
                logger.info(f"🔄 Cache refreshed: {len(memory_contents)} embeddings in {duration:.2f}s")

        except Exception as e:
            raise RuntimeError(f"🧹 Failed to refresh cache for user {user_id} after {(time.time() - start_time):.2f}s: {str(e)}")

    async def _execute_single_operation(self, operation: Models.MemoryOperation, user: Any) -> str:
        """Execute a single memory operation."""
        content = operation.content.strip()
        memory_id = operation.id.strip()

        if operation.operation == Models.MemoryOperationType.CREATE:
            if not content:
                return Models.OperationResult.SKIPPED_EMPTY_CONTENT.value
            await asyncio.wait_for(
                asyncio.to_thread(Memories.insert_new_memory, user.id, content),
                timeout=Constants.DATABASE_OPERATION_TIMEOUT_SEC,
            )
            return Models.MemoryOperationType.CREATE.value

        elif operation.operation == Models.MemoryOperationType.UPDATE:
            if not memory_id:
                return Models.OperationResult.SKIPPED_EMPTY_ID.value
            if not content:
                return Models.OperationResult.SKIPPED_EMPTY_CONTENT.value
            await asyncio.wait_for(
                asyncio.to_thread(Memories.update_memory_by_id_and_user_id, memory_id, user.id, content),
                timeout=Constants.DATABASE_OPERATION_TIMEOUT_SEC,
            )
            return Models.MemoryOperationType.UPDATE.value

        elif operation.operation == Models.MemoryOperationType.DELETE:
            if not memory_id:
                return Models.OperationResult.SKIPPED_EMPTY_ID.value
            await asyncio.wait_for(
                asyncio.to_thread(Memories.delete_memory_by_id_and_user_id, memory_id, user.id),
                timeout=Constants.DATABASE_OPERATION_TIMEOUT_SEC,
            )
            return Models.MemoryOperationType.DELETE.value

        return Models.OperationResult.UNSUPPORTED.value

    def _remove_refs_from_schema(self, schema: Dict[str, Any], schema_defs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Remove $ref references and ensure required fields for Azure OpenAI."""
        if not isinstance(schema, dict):
            return schema

        if "$ref" in schema:
            ref_path = schema["$ref"]
            if ref_path.startswith("#/$defs/"):
                def_name = ref_path.split("/")[-1]
                if schema_defs and def_name in schema_defs:
                    return self._remove_refs_from_schema(schema_defs[def_name].copy(), schema_defs)
            return {"type": "object"}

        result = {}
        for key, value in schema.items():
            if key == "$defs":
                continue
            elif isinstance(value, dict):
                result[key] = self._remove_refs_from_schema(value, schema_defs)
            elif isinstance(value, list):
                result[key] = [(self._remove_refs_from_schema(item, schema_defs) if isinstance(item, dict) else item) for item in value]
            else:
                result[key] = value

        if result.get("type") == "object" and "properties" in result:
            result["required"] = list(result["properties"].keys())

        return result

    async def _query_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Optional[BaseModel] = None,
    ) -> Union[str, BaseModel]:
        """Query OpenWebUI's internal model system with Pydantic model parsing."""
        if not hasattr(self, "__request__") or not hasattr(self, "__user__"):
            raise RuntimeError("🔧 Pipeline interface not properly initialized. __request__ and __user__ required.")

        model_to_use = self.__model__
        if not model_to_use:
            raise ValueError("🤖 No model specified for LLM operations")

        form_data = {
            "model": model_to_use,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 4096,
            "stream": False,
        }

        if response_model:
            raw_schema = response_model.model_json_schema()
            schema_defs = raw_schema.get("$defs", {})
            schema = self._remove_refs_from_schema(raw_schema, schema_defs)
            schema["type"] = "object"
            form_data["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": schema,
                },
            }

        try:
            response = await asyncio.wait_for(
                generate_chat_completion(
                    self.__request__,
                    form_data,
                    user=await asyncio.to_thread(Users.get_user_by_id, self.__user__["id"]),
                ),
                timeout=Constants.LLM_CONSOLIDATION_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"⏱️ LLM query timed out after {Constants.LLM_CONSOLIDATION_TIMEOUT_SEC}s")
        except Exception as e:
            raise RuntimeError(f"🤖 LLM query failed: {str(e)}")

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
