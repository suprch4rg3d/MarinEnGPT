import os
import requests
from typing import Optional, Dict
import openai
import chainlit as cl
from helpers import *

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine import SimpleMultiModalQueryEngine
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.evaluation import (
    RetrieverEvaluator,
    QueryResponseEvaluator,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.embeddings import resolve_embed_model
# from llama_index.core.utils import cosine_similarity  - Removed..why LlamaIndex, why?
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from llama_index.core.schema import NodeWithScore  

from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings

from ocr_tesseract import ocr_generator
from image_extractor import extract_images_from_pdf
from weather import weather_keywords, detect_tense_simple
from llama_index.tools.weather import OpenWeatherMapToolSpec

from textwrap import shorten
from collections import defaultdict, Counter
import re
import json
import asyncio
from datetime import datetime
import csv
from statistics import mean
import zipfile
import glob
from pathlib import Path
from decimal import Decimal  # Retrieval Metrics are not int or float...
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)


# API KEYS
ENABLE_USER_ENV = os.getenv("ENABLE_USER_ENV", "false").lower() == "true"
OPENAI_TOKEN: Optional[str] = os.getenv("OPENAI_API_KEY")
HF_TOKEN: Optional[str] = os.getenv("HUGGING_FACE_TOKEN")
OPENWEATHER_TOKEN: Optional[str] = os.getenv("OPENWEATHER_API_KEY")

TOP_K_RESULTS = int(os.getenv("TOP_K_RETRIEVAL_RESULTS", 5))

# Decoupled Retrieval & Synthesis Configuration
ENABLE_DECOUPLED_SYNTHESIS = (
    os.getenv("ENABLE_DECOUPLED_SYNTHESIS", "false").lower() == "true"
)
CHUNK_GROUP_FIELD = os.getenv("CHUNK_GROUP_FIELD", "parent_id")
CHUNK_SYNTHESIS_MIN = int(os.getenv("CHUNK_SYNTHESIS_MIN", 1))
CHUNK_SYNTHESIS_MAX = int(os.getenv("CHUNK_SYNTHESIS_MAX", 6))
CHUNK_SYNTHESIS_ORDERED = os.getenv("CHUNK_SYNTHESIS_ORDERED", "true").lower() == "true"

# Evaluation Logging Configuration
ENABLE_SESSION_EVAL = os.getenv("ENABLE_SESSION_EVAL", "false").lower() == "true"
EVAL_FOLDER = os.getenv("EVAL_OUTPUT_DIR", "./evals")
EVAL_TRIGGER_KEYWORD = os.getenv("EVAL_TRIGGER_KEYWORD", "!freeze_eval").strip().lower()
EVAL_MAX_LOGGED_QUERIES = int(os.getenv("EVAL_MAX_LOGGED_QUERIES", 0))  # 0 = unlimited
EVAL_ZIP_EXPORT = os.getenv("EVAL_ZIP_EXPORT", "false").lower() == "true"

# Evaluation Trigger Commands
EVAL_REPLAY_ENABLED = os.getenv("EVAL_REPLAY_ENABLED", "true").lower() == "true"
EVAL_REPLAY_TOP_K = int(os.getenv("EVAL_REPLAY_TOP_K", 3))
EVAL_REPLAY_TRIGGER = os.getenv("EVAL_REPLAY_TRIGGER", "!replay_eval").strip().lower()

# Evaluation List Settings
EVAL_LIST_ENABLED = os.getenv("EVAL_LIST_ENABLED", "true").lower() == "true"
EVAL_LIST_TRIGGER = os.getenv("EVAL_LIST_TRIGGER", "!list_evals").strip().lower()

# Evaluation Comparison Settings
EVAL_COMPARE_ENABLED = os.getenv("EVAL_COMPARE_ENABLED", "true").lower() == "true"
EVAL_COMPARE_TRIGGER = (
    os.getenv("EVAL_COMPARE_TRIGGER", "!compare_eval").strip().lower()
)

# Evaluation Listing Interaction
EVAL_LIST_TRIGGER = os.getenv("EVAL_LIST_TRIGGER", "!list_evals").strip().lower()
EVAL_LIST_LIMIT = int(os.getenv("EVAL_LIST_LIMIT", 10))
EVAL_LIST_FORMAT = os.getenv("EVAL_LIST_FORMAT", "both").lower()

# Optional filters for QA dataset generation
ENABLE_QA_DATASET_GENERATION = (
    os.getenv("ENABLE_QA_DATASET_GENERATION", "false").lower() == "true"
)
QA_NUM_QUESTIONS = int(os.getenv("QA_NUM_QUESTIONS", 1))
QA_SOURCE_FILE_FILTER = os.getenv("QA_SOURCE_FILE_FILTER", "").strip() or None
QA_EMBEDDING_MODE_FILTER = os.getenv("QA_EMBEDDING_MODE_FILTER", "").strip() or None
QA_MATCH_THRESHOLD = float(os.getenv("QA_MATCH_THRESHOLD", 0.85))
RETRIEVAL_DEBUG_MODE = os.getenv("RETRIEVAL_DEBUG", "false").lower() == "true"

def debug_log(*args, **kwargs):
    if RETRIEVAL_DEBUG_MODE:
        print(*args, **kwargs)

def load_embedding_config(config_path="./embeddings_openai_persistence.json"):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            model = cfg.get("model", "text-embedding-3-small")
            dimensions = cfg.get("dimensions")
            use_dimensions = cfg.get("use_dimensions", False)
            return model, dimensions if use_dimensions else None
    except Exception as e:
        print(f"\033[91m[ERROR]\033[0m Failed to load config.json: {e}")
        return "text-embedding-3-small", None


# Set default LLM and embedding model
llm = OpenAI(model="gpt-4o", temperature=0.0)

model_name, custom_dims = load_embedding_config()
embed_model = OpenAIEmbedding(
    model=model_name,
    dimensions=custom_dims,
)

Settings.llm = llm
Settings.embed_model = embed_model


def load_last_chroma_collection_config(
    config_path="./embeddings_openai_persistence.json",
):
    """
    Loads the last used ChromaDB collection and persistence path from the CLI config file.

    Args:
        config_path (str): Path to the CLI config file of Embeddings using OpenAI API utility (default: embeddings_openai_persistence.json)

    Returns:
        dict: {
            "chromadb_collection_name": str or None,
            "chromadb_persistence_path": str
        }
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return {
            "chromadb_collection_name": config.get("chromadb_collection_name"),
            "chromadb_persistence_path": config.get(
                "chromadb_persistence_path", "./data/chromadb"
            ),
        }
    except Exception as e:
        print(f"\033[91m[ERROR]\033[0m Failed to load CLI config: {e}")
        return {}


# Path and collection for ChromaDB
cli_chroma_config = load_last_chroma_collection_config()
CHROMA_PATH = cli_chroma_config.get("chromadb_persistence_path", "./data/chromadb")
CHROMA_COLLECTION = cli_chroma_config.get(
    "chromadb_collection_name", "MarineEngineeringManuals"
)


# Connect to ChromaDB and load pre-generated index
def load_chromadb_index():
    print("\033[94m[INFO]\033[0m Connecting to ChromaDB...")

    # Load embedding model + dimension config
    model_name, expected_dimensions = load_embedding_config()

    client = PersistentClient(
        path=CHROMA_PATH,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    collections = client.list_collections()
    if CHROMA_COLLECTION not in collections:
        raise ValueError(
            f"\033[91m[ERROR]\033[0m Collection '{CHROMA_COLLECTION}' not found in {CHROMA_PATH}."
        )

    chroma_collection = client.get_collection(name=CHROMA_COLLECTION)
    result = chroma_collection.get()
    doc_ids = result["ids"]
    doc_metadatas = result.get("metadatas", [])
    doc_texts = result.get("documents", [])
    doc_count = len(doc_ids)

    # Check dimensionality mismatch
    if expected_dimensions:
        actual_dim = chroma_collection.metadata.get("dimension")
        if actual_dim and actual_dim != expected_dimensions:
            print(
                f"\033[93m[WARNING]\033[0m Embedding dimension mismatch: "
                f"ChromaDB = {actual_dim}, Config = {expected_dimensions}\n"
            )

    # Check for duplicate custom_ids
    duplicates = [item for item, count in Counter(doc_ids).items() if count > 1]
    if duplicates:
        print(
            f"\033[93m[WARNING]\033[0m Duplicate custom_id(s) found: "
            f"{', '.join(duplicates[:5])}{'...' if len(duplicates) > 5 else ''}\n"
        )

    # Check for missing or incomplete metadata
    missing_meta = 0
    for md in doc_metadatas:
        if not md or not isinstance(md, dict):
            missing_meta += 1
        elif not all(key in md for key in ["embedding_mode", "model"]):
            missing_meta += 1
    if missing_meta > 0:
        print(
            f"\033[93m[WARNING]\033[0m {missing_meta} entries have missing or incomplete metadata.\n"
        )

    # Check for all document fields empty
    empty_doc_count = sum(1 for doc in doc_texts if not doc or not doc.strip())
    warning_flag = empty_doc_count == len(doc_texts) and doc_count > 0

    # --- Group Overview ---
    print(
        f"\033[92m[SUCCESS]\033[0m Connected to collection: '{CHROMA_COLLECTION}'"
        + (" \033[91m[!]\033[0m" if warning_flag else "")
    )
    print(f"\033[90m ├── Documents loaded:\033[0m {doc_count}")

    # Group and print preview
    show_group_preview = os.getenv("GROUP_PREVIEW_ENABLED", "true").lower() == "true"
    group_preview_limit = int(os.getenv("GROUP_PREVIEW_LIMIT", 5))

    if show_group_preview:
        group_stats = defaultdict(
            lambda: {"pages": set(), "chunks": set(), "modes": set()}
        )
        orphaned_chunks = set()
        all_ids_set = set(doc_ids)

        for idx, doc_id in enumerate(doc_ids):
            metadata = doc_metadatas[idx] if idx < len(doc_metadatas) else {}

            match = re.match(r"^(.*)_page_(\d+)(?:_chunk_(\d+))?$", doc_id)
            if match:
                group_id = match.group(1)
                page_id = f"{group_id}_page_{match.group(2)}"
                is_chunk = match.group(3) is not None
            else:
                group_id = doc_id.split("_page_")[0]
                page_id = doc_id
                is_chunk = "_chunk_" in doc_id

            embedding_mode = metadata.get("embedding_mode", "unknown")
            parent_id = metadata.get("parent_id")

            if is_chunk:
                group_stats[group_id]["chunks"].add(doc_id)
                group_stats[group_id]["pages"].add(page_id)
                group_stats[group_id]["modes"].add("chunked")

                # Check if parent exists only in mixed mode
                if "full" in group_stats[group_id]["modes"]:
                    if parent_id and parent_id not in all_ids_set:
                        orphaned_chunks.add(doc_id)
            else:
                group_stats[group_id]["pages"].add(page_id)
                group_stats[group_id]["modes"].add("full")

        print(f"\033[90m ├── Document Groups:\033[0m {len(group_stats)}")
        for idx, (group_id, stats) in enumerate(sorted(group_stats.items())):
            if idx >= group_preview_limit:
                break
            pages = len(stats["pages"])
            chunks = len(stats["chunks"])
            mode = "+".join(sorted(stats["modes"]))
            print(f"     ├─ {group_id}  ({pages} pages, {chunks} chunks, mode: {mode})")

        if len(group_stats) > group_preview_limit:
            print(
                f"     └─ ...and {len(group_stats) - group_preview_limit} more groups."
            )

        if orphaned_chunks:
            print(
                f"\033[93m[WARNING]\033[0m Found {len(orphaned_chunks)} orphaned chunks with missing parent_id links.\n"
            )

    print(f"\033[90m └── Vector DB path:\033[0m {CHROMA_PATH}\n")

    if warning_flag:
        print(
            "\033[93m[WARNING]\033[0m All document fields are empty strings. "
            "You may have loaded embeddings without associated text. "
            "Synthesis or semantic responses may be incomplete.\n"
        )

    if doc_count == 0:
        print(
            "\033[93m[WARNING]\033[0m Collection is empty. Semantic queries will be skipped.\n"
        )
        return None

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, storage_context=storage_context
    )
    return index


index = load_chromadb_index()


@retry(
    wait=wait_random_exponential(min=1, max=60),  # exponential backoff between retries
    stop=stop_after_attempt(6),  # give up after 6 failed attempts
    retry=retry_if_exception_type(
        (openai.RateLimitError, openai.APIError, openai.Timeout)
    ),
)
def safe_generate_dataset(generator):
    """
    Safely generates a QA dataset using RagDatasetGenerator with built-in retry logic.

    This function wraps `generate_dataset_from_nodes()` with automatic retries
    to handle transient errors from the OpenAI API, such as rate limits,
    timeouts, or server-side failures.

    Retries use exponential backoff with jitter (random wait between 1 and 60 seconds),
    and will give up after 6 failed attempts.

    Args:
        generator (RagDatasetGenerator): The QA dataset generator instance.

    Returns:
        List[GeneratedQuestionAnswerPair]: List of generated QA pairs.

    Raises:
        openai.error.OpenAIError: If the operation fails even after retries.
    """
    return generator.generate_dataset_from_nodes()


def generate_qa_dataset(
    num_questions=1,
    embedding_mode_filter: Optional[str] = None,
    source_file_filter: Optional[str] = None,
):
    """
    Generate a synthetic QA dataset from ChromaDB-stored nodes using LlamaIndex's RagDatasetGenerator.

    This function retrieves all nodes from the currently loaded ChromaDB index and applies optional filters
    to restrict the dataset to specific types of content (e.g., only chunked embeddings or specific manuals).
    It constructs document nodes and generates synthetic QA pairs using RagDatasetGenerator.

    Each QA pair contains a question and its expected reference chunk IDs for retrieval evaluation,
    such as hit rate, precision, recall, MRR, etc.

    Args:
        num_questions (int): Number of synthetic questions to generate per chunk.
        embedding_mode_filter (str, optional): If provided, only includes nodes with this embedding mode
                                               (e.g., "chunked" or "full").
        source_file_filter (str, optional): If provided, only includes nodes whose 'source_file' metadata
                                            contains this string (e.g., a manual code like "MF-194").

    Returns:
        List[GeneratedQuestionAnswerPair]: A list of QA entries with queries and expected reference chunks.
    """
    if not ENABLE_QA_DATASET_GENERATION:
        print(
            "\033[94m[INFO]\033[0m Synthetic QA dataset generation is currently disabled.\n"
            "This dataset provides ground truth answers needed to evaluate retrieval metrics "
            "(Hit Rate, MRR, Precision, Recall, NDCG). Without it, these metrics cannot be computed.\n\n"
            "To enable generation, set `ENABLE_QA_DATASET_GENERATION=true` in your environment variables. "
            "You can also use `QA_SOURCE_FILE_FILTER` or `QA_EMBEDDING_MODE_FILTER` to limit the scope."
        )
        return []

    if index is None:
        raise RuntimeError(
            "Index is not loaded. Ensure ChromaDB collection is connected."
        )

    # Access nodes directly from ChromaVectorStore
    vector_store = getattr(index, "_vector_store", None)
    if not isinstance(vector_store, ChromaVectorStore):
        raise RuntimeError("This QA generator only supports ChromaVectorStore.")

    # Retrieve all node ids and then nodes from the ChromaDB index
    all_node_ids = vector_store._collection.get()["ids"]
    all_nodes = vector_store.get_nodes(all_node_ids)
    # dd(all_nodes)

    # TEMP DEBUGGING BLOCK:
    print(f"\033[90m[DEBUG]\033[0m Total nodes retrieved from ChromaVectorStore: {len(all_nodes)}")
    for i, node in enumerate(all_nodes[:3]):  # Print first 3 nodes
        print(f"\033[90m[DEBUG]\033[0m Node {i} metadata: {node.metadata}")

    # Apply optional filters
    filtered_nodes = []
    for node in all_nodes:
        metadata = node.metadata or {}

        if (
            embedding_mode_filter
            and metadata.get("embedding_mode") != embedding_mode_filter
        ):
            continue

        source_field = metadata.get("source_file", "")

        # Fallback: infer from custom_id if missing (for full mode)
        if not source_field:
            custom_id = getattr(node, "node_id", "") or getattr(node, "id_", "")
            if custom_id:
                base = custom_id.split("_page_")[0].strip()
                if base:
                    source_field = f"{base}.md"

        # Use substring matching for source_file_filter
        if (
            source_file_filter
            and source_file_filter.lower() not in source_field.lower()
        ):
            continue

        filtered_nodes.append(node)
        # dd(filtered_nodes)

    print(
        f"\033[90m[DEBUG]\033[0m Using {len(filtered_nodes)} filtered nodes from ChromaDB for QA generation."
    )

    if not filtered_nodes:
        raise RuntimeError("No nodes matched the specified filters.")

    generator = RagDatasetGenerator(
        nodes=filtered_nodes,
        num_questions_per_chunk=num_questions,
    )
    return safe_generate_dataset(generator)


def find_closest_qa_query(
    user_query: str, qa_lookup: dict, threshold: Optional[float] = None
):
    """
    Find the closest matching query in the QA lookup using semantic similarity.

    This function compares the user's query against the keys in `qa_lookup` by
    computing cosine similarity between their OpenAI embeddings. It selects the
    most semantically similar QA question above a given threshold.

    Reason:
        The Retriever Evaluator of LlamaIndex works by comparing retrieved chunk IDs
        against ground-truth expected IDs derived from a synthetic QA dataset,
        which are keyed by exact question strings.

        However, exact string matching is too brittle for synthetic QA evaluation.
        The user's phrasing is unlikely to match generated questions exactly.
        Semantic (fuzzy) matching ensures we can align the user's intent with
        the most relevant synthetic QA example to evaluate retrieval accuracy,
        even when the query is paraphrased or reworded.

        This approach improves robustness of evaluation metrics like Hit Rate,
        MRR, Precision, Recall, and NDCG — especially when working with
        small benchmark manuals like MF-194 that help avoid OpenAI rate limits.

    Args:
        user_query (str): The query submitted by the user.
        qa_lookup (dict): Mapping of QA questions to expected node IDs.
        threshold (float, optional): Minimum cosine similarity required for a match.
                                     If None, uses QA_MATCH_THRESHOLD from .env.


    Returns:
        tuple[str, list[str]]: The matched QA question and its expected IDs,
                               or (None, None) if no match meets the threshold.
    """
    if not qa_lookup:
        return None, None

    if threshold is None:
        threshold = QA_MATCH_THRESHOLD

    user_vec = embed_model.get_text_embedding(user_query)
    user_vec = np.array(user_vec).reshape(1, -1)

    best_match = None
    best_score = -1

    print("\n\033[90m[DEBUG]\033[0m Cosine similarity scores to each QA:")
    for known_query in qa_lookup.keys():
        known_vec = embed_model.get_text_embedding(known_query)
        known_vec = np.array(known_vec).reshape(1, -1)
        score = cosine_similarity(user_vec, known_vec)[0][0]
        print(f" → {score:.4f} | {known_query[:80]}...")
        if score > best_score:
            best_score = score
            best_match = known_query
    
    print(f"\033[90m[DEBUG]\033[0m Best match score: {best_score:.4f}")
    print(f"\033[90m[DEBUG]\033[0m Best matched QA: {best_match}")

    if best_score >= threshold:
        return best_match, qa_lookup[best_match]
    return None, []


# Log top-k results with score and metadata
def log_top_k_results(nodes, k=5):
    ANSI_BLUE = "\033[94m"
    ANSI_GREEN = "\033[92m"
    ANSI_GRAY = "\033[90m"
    ANSI_RESET = "\033[0m"

    print(
        f"\n{ANSI_BLUE}[INFO]{ANSI_RESET} Top {min(k, len(nodes))} Retrieved Chunks:\n"
    )

    for idx, node in enumerate(nodes[:k], 1):
        score = getattr(node, "score", "N/A")
        node_id = getattr(node, "id_", "Unknown")
        metadata = node.metadata or {}
        model = metadata.get("model", "N/A")
        tokens = metadata.get("token_count", "N/A")
        source_file = metadata.get("source_file", "N/A")
        snippet = shorten(str(getattr(node, "text", "")), width=100, placeholder="...")

        print(f"{ANSI_GREEN}[Node {idx}]{ANSI_RESET}")
        print(f" ├─ ID: {node_id}")
        print(f" ├─ Score: {score}")
        print(f" ├─ Model: {model}")
        print(f" ├─ Tokens: {tokens}")
        print(f" ├─ File: {source_file}")
        print(f' └─ Content Snippet: {ANSI_GRAY}"{snippet}"{ANSI_RESET}\n')

        # Debug content comparison: fallback from metadata["document"] or metadata["documents"]
        doc_fallback = metadata.get("document") or metadata.get("documents")
        if not getattr(node, "text", "").strip():
            print(
                f" └─ Content Snippet (metadata.document): {ANSI_GRAY}[EMPTY]{ANSI_RESET}\n"
            )
        else:
            doc_snippet = shorten(doc_fallback or "", width=100, placeholder="...")
            print(
                f' └─ Content Snippet (metadata.document): {ANSI_GRAY}"{doc_snippet}"{ANSI_RESET}\n'
            )


def summarize_eval_data(eval_data):
    """
    Computes key statistics, response quality metrics, and averaged retrieval metrics
    from the evaluation dataset.

    Returns:
        dict: Summary including average top-k score, most common elements,
              averaged retriever metrics (hit_rate, precision, etc.),
              and response evaluation metrics.
    """
    summary = {
        "total_queries": len(eval_data),
        "average_top_k_score": 0.0,
        "average_response_score": None,
        "response_pass_rate": None,
        "common_response_feedback": [],
        "most_common_chunks": [],
        "most_common_sources": [],
        "most_common_models": [],
        "matched_manual_queries": 0,
        "unmatched_manual_queries": 0,
    }

    scores = []
    response_scores = []
    response_passes = []
    feedback_counter = Counter()
    chunk_counter = Counter()
    source_counter = Counter()
    model_counter = Counter()

    metrics_accumulator = {
        "hit_rate": [],
        "mrr": [],
        "precision": [],
        "recall": [],
        "ap": [],
        "ndcg": [],
    }

    for entry in eval_data:
        if entry.get("matched_manual") is True:
            summary["matched_manual_queries"] += 1
        else:
            summary["unmatched_manual_queries"] += 1
            
        for result in entry.get("top_k_results", []):
            try:
                score = float(result.get("score", 0))
                scores.append(score)
            except ValueError:
                continue

            chunk_id = result.get("id")
            source_file = result.get("source_file", "N/A")
            model = result.get("model", "N/A")

            chunk_counter[chunk_id] += 1
            source_counter[source_file] += 1
            model_counter[model] += 1

        # Aggregate retrieval metrics
        retrieval_metrics = entry.get("retriever_metrics", {})
        if isinstance(retrieval_metrics, dict):
            for metric in metrics_accumulator.keys():
                val = retrieval_metrics.get(metric)
                if isinstance(val, (float, int, Decimal)):
                    metrics_accumulator[metric].append(val)

        # Aggregate response quality metrics
        response = entry.get("response_quality", {})
        if isinstance(response, dict):
            score = response.get("score")
            if isinstance(score, (float, int)):
                response_scores.append(score)

            passed = response.get("passing")
            if isinstance(passed, bool):
                response_passes.append(passed)

            feedback = response.get("feedback")
            if feedback:
                feedback_counter[feedback] += 1

    # Compute summary stats
    if scores:
        summary["average_top_k_score"] = round(mean(scores), 4)

    if response_scores:
        summary["average_response_score"] = round(mean(response_scores), 4)

    if response_passes:
        pass_rate = sum(response_passes) / len(response_passes)
        summary["response_pass_rate"] = round(pass_rate, 4)

    if feedback_counter:
        summary["common_response_feedback"] = feedback_counter.most_common(3)

    for metric, values in metrics_accumulator.items():
        if values:
            summary[f"average_{metric}"] = round(mean(values), 4)

    summary["most_common_chunks"] = chunk_counter.most_common(5)
    summary["most_common_sources"] = source_counter.most_common(5)
    summary["most_common_models"] = model_counter.most_common(5)

    return summary


def group_chunks_for_synthesis(
    nodes, group_field="parent_id", min_size=1, max_size=6, ordered=True
):
    grouped = defaultdict(list)
    for node in nodes:
        group_id = node.metadata.get(group_field)
        if group_id:
            grouped[group_id].append(node)

    synthesis_units = []
    for group_id, chunks in grouped.items():
        if not (min_size <= len(chunks) <= max_size):
            continue
        if ordered:
            chunks.sort(key=lambda x: x.metadata.get("chunk_index", 0))
        merged_text = "\n".join(chunk.text for chunk in chunks if chunk.text)
        synthesis_units.append((group_id, merged_text))

    return synthesis_units


class LLMResponseWrapper:
    """
    Simple wrapper to make a plain string `response_text` compatible with LlamaIndex's
    response evaluators, which expect a `.response` attribute on the response object.

    This is useful when you are using raw LLM completions (like OpenAI.complete())
    and want to evaluate the response using:
        - QueryResponseEvaluator.evaluate_response()
        - FaithfulnessEvaluator.evaluate_response()
        - AnswerRelevancyEvaluator.evaluate_response()

    Example:
        wrapper = LLMResponseWrapper("This is a response.")
        evaluator.evaluate_response(query, wrapper)
    """

    def __init__(self, response: str, source_nodes=None):
        self.response = response
        self.source_nodes = source_nodes or []


@cl.on_chat_start
async def start():
    print(f"\033[90m[DEBUG]\033[0m Top-K retrieval log threshold: {TOP_K_RESULTS}")

    # Initialize evaluation dataset for the session
    cl.user_session.set("eval_data", [])
    cl.user_session.set("eval_start_time", datetime.now().isoformat())

    # Initialize placeholder for synthetic QA lookup - Will be lazily populated on first semantic query if benchmarking is needed.
    cl.user_session.set("qa_lookup", None)

    if ENABLE_USER_ENV:
        # Retrieve user-specific environment variables
        user_env = cl.user_session.get("env")
        openai_token = user_env.get("OPENAI_API_KEY")

        if not openai_token:
            await cl.Message(
                content="No OpenAI API key was provided. Please restart and enter your key to proceed."
            ).send()
            return
    else:
        # Retrieve project environment variables
        openai_token = os.getenv("OPENAI_API_KEY")

    # Store in session for downstream use
    cl.user_session.set("openai_token", openai_token)

    # Set up models dynamically
    llm = OpenAI(model="gpt-4o", temperature=0.0, api_key=openai_token)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=openai_token)

    Settings.llm = llm
    Settings.embed_model = embed_model

    # Check for short-term context toggle
    enable_context = os.getenv("ENABLE_CHAT_HISTORY_CONTEXT", "false").lower() == "true"
    cl.user_session.set("enable_chat_history", enable_context)

    if enable_context:
        cl.user_session.set("chat_history", [])  # Initialize empty list

    await cl.Message(
        author="MarinEnGPT",
        content="Hello! I am MarinEnGPT, an AI Assistant specialized in Marine Engineering Service Manuals.\n\nHow may I help you?",
    ).send()


@cl.action_callback("view_as_image")
async def view_as_image(action: cl.Action):
    """Callback to display references as images."""
    references = action.payload.get("references", [])
    IMAGE_BASE_PATH = os.getenv("IMAGE_BASE_PATH", "./data/images/")

    seen_pages = set()
    elements = []

    for idx, ref in enumerate(references, 1):
        node_id = ref.get("id")

        # Normalize to base page ID: strip "_chunk_#" if present
        base_id = re.sub(r"_chunk_\d+$", "", node_id)
        if base_id in seen_pages:
            continue  # Skip duplicates
        seen_pages.add(base_id)

        # Path: ./data/images/{group_id}/{base_id}.png
        group_id = base_id.split("_page_")[0]
        image_path = os.path.join(IMAGE_BASE_PATH, group_id, f"{base_id}.png")

        if os.path.exists(image_path):
            elements.append(
                cl.Image(
                    path=image_path,
                    name=f"Reference Image {idx}",
                    display="inline",
                    size="medium",
                )
            )
        else:
            await cl.Message(
                content=f"Warning: Image for reference {idx} not found at {image_path}."
            ).send()

    if elements:
        await cl.Message(
            content="Here are the references as images:", elements=elements
        ).send()
    else:
        await cl.Message(content="No images available for the selected references.").send()


@cl.action_callback("view_as_markdown")
async def view_as_markdown(action: cl.Action):
    """Callback to display references as markdown."""
    references = action.payload.get("references", [])

    for idx, ref in enumerate(references, 1):
        source_file = ref.get("source_file", "Unknown source")
        document_snippet = ref.get("text", "")[:500]  # Adjust snippet length as needed

        reference_message = f"**Reference {idx}:**\n\n"
        reference_message += f"Source: `{source_file}`\n\n"
        reference_message += f'Snippet: "{document_snippet}..."\n\n'

        await cl.Message(content=reference_message).send()


@cl.action_callback("skip_references")
async def skip_references(action: cl.Action):
    """Callback to handle skipping references."""
    await cl.Message(content="References have been skipped.").send()


def format_eval_summary(summary: dict) -> list[str]:
    lines = [
        f"Total Queries: {summary.get('total_queries', 0)}",
        f"Matched Manual Queries: {summary.get('matched_manual_queries', 0)}",
        f"Unmatched Manual Queries: {summary.get('unmatched_manual_queries', 0)}",
        f"Average Top-K Score: {summary.get('average_top_k_score', 0.0)}",
    ]

    if summary.get("average_response_score") is not None:
        lines.append(f"Average Response Score: {summary['average_response_score']}")

    if summary.get("response_pass_rate") is not None:
        percent = round(summary["response_pass_rate"] * 100, 2)
        lines.append(f"Response Pass Rate: {percent}%")

    if summary.get("common_response_feedback"):
        lines.append("Most Common Feedback:")
        for fb, count in summary["common_response_feedback"]:
            lines.append(f" - {fb} ({count}x)")

    # Add retrieval metrics if they exist
    metric_keys = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]
    has_metrics = any(f"average_{k}" in summary for k in metric_keys)

    if has_metrics:
        lines.append("\nAverage Retrieval Metrics:")
        for metric in metric_keys:
            key = f"average_{metric}"
            if key in summary:
                lines.append(f" - {metric.upper()}: {summary[key]}")

    lines.append("\nTop 5 Retrieved Chunk IDs:")
    for chunk_id, count in summary.get("most_common_chunks", []):
        lines.append(f" - {chunk_id} ({count} hits)")

    lines.append("\nTop 5 Source Files:")
    for src, count in summary.get("most_common_sources", []):
        lines.append(f" - {src} ({count} hits)")

    lines.append("\nTop 5 Embedding Models:")
    for model, count in summary.get("most_common_models", []):
        lines.append(f" - {model} ({count} hits)")

    return lines


# Evaluation Features
#
# Evaluation Trigger: List available evaluation log files
async def handle_eval_list_trigger(content: str) -> bool:
    # Check if the message matches the trigger and listing is enabled
    if content.strip().lower() == EVAL_LIST_TRIGGER and EVAL_LIST_ENABLED:
        # Find all matching evaluation log files
        eval_files = sorted(
            Path(EVAL_FOLDER).rglob("eval_log_*.json"),  # or "*.csv" or both
            key=os.path.getmtime,
            reverse=True,
        )

        # If no logs found, notify user
        if not eval_files:
            await cl.Message(
                author="Evaluation Bot", content="No saved evaluation logs found."
            ).send()
            return True

        # Limit how many logs to show
        display_files = eval_files[:EVAL_LIST_LIMIT]

        # Prepare message with timestamps
        msg_lines = [f"Showing latest {len(display_files)} evaluation logs:"]
        for path in display_files:
            fname = os.path.basename(path)
            mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime(
                "%Y-%m-%d %H:%M"
            )
            msg_lines.append(f"- `{fname}`  (last modified: {mtime})")

        # Add replay hint
        msg_lines.append("\nTo replay a log, type: `!replay_eval filename.json`")

        # Send to Chainlit
        await cl.Message(author="Evaluation Bot", content="\n".join(msg_lines)).send()

        return True  # Trigger matched and handled

    return False  # Not a match


# Evaluation Trigger: Replay a saved evaluation log file
async def handle_eval_replay_trigger(content: str) -> bool:
    # Check if the message starts with the replay trigger and the feature is enabled
    if (
        content.strip().lower().startswith(EVAL_REPLAY_TRIGGER + " ")
        and EVAL_REPLAY_ENABLED
    ):
        # Extract the filename from the message
        filename = content.strip().split(" ", 1)[-1]
        matches = list(Path(EVAL_FOLDER).rglob(filename))
        if not matches:
            await cl.Message(
                author="Evaluation Bot",
                content=f"File `{filename}` not found in `{EVAL_FOLDER}`.",
            ).send()
            return True

        path = str(matches[0])  # Use the first match

        try:
            # Load the evaluation data from the file
            with open(path, "r", encoding="utf-8") as f:
                replay_data = json.load(f)

            # Prepare message content with summary and top 3 results per query
            lines = [
                f"Replay of `{filename}`",
                f"Total Queries: {len(replay_data)}",
                "",
            ]
            for idx, entry in enumerate(replay_data, 1):
                lines.append(f"{idx}. {entry['query']} ({entry['timestamp']})")

                # Display top 3 retrieved chunks (if available)
                for chunk in entry.get("top_k_results", [])[:3]:
                    lines.append(
                        f"   - ID: {chunk.get('id')} | Score: {chunk.get('score')} | Source: {chunk.get('source_file')}"
                    )
                lines.append("")  # spacing between entries

            # Send the assembled log to Chainlit
            await cl.Message(author="Evaluation Bot", content="\n".join(lines)).send()

        except Exception as e:
            # Handle unexpected errors (e.g., malformed JSON)
            await cl.Message(
                author="Evaluation Bot", content=f"Failed to load `{filename}`: {e}"
            ).send()

        return True  # Trigger was matched and handled

    return False  # Trigger was not matched


# Evaluation Trigger: Compare Two Evaluation Logs
async def handle_eval_compare_trigger(content: str) -> bool:
    """
    Handles the evaluation comparison trigger (!compare_eval file1.json file2.json).
    Returns True if this trigger was matched and handled.
    """
    if (
        content.strip().lower().startswith(EVAL_COMPARE_TRIGGER)
        and EVAL_COMPARE_ENABLED
    ):
        tokens = content.strip().split()
        if len(tokens) != 3:
            await cl.Message(
                author="Evaluation Bot",
                content=f"Usage: `{EVAL_COMPARE_TRIGGER} file1.json file2.json`",
            ).send()
            return True

        file1, file2 = tokens[1], tokens[2]

        def find_eval_file(filename):
            for root, _, files in os.walk(EVAL_FOLDER):
                if filename in files:
                    return os.path.join(root, filename)
            return None

        path1 = find_eval_file(file1)
        path2 = find_eval_file(file2)

        # Check if both files were found
        if not path1 or not path2:
            await cl.Message(
                author="Evaluation Bot",
                content=f"One or both files could not be found in `{EVAL_FOLDER}` (including subfolders).",
            ).send()
            return True

        try:
            # Load both evaluation logs
            with open(path1, "r", encoding="utf-8") as f1, open(
                path2, "r", encoding="utf-8"
            ) as f2:
                eval1 = json.load(f1)
                eval2 = json.load(f2)

            # Map queries to entries
            q1 = {entry["query"]: entry for entry in eval1}
            q2 = {entry["query"]: entry for entry in eval2}

            # Identify shared and unique queries
            shared_queries = set(q1) & set(q2)
            only_in_1 = set(q1) - set(q2)
            only_in_2 = set(q2) - set(q1)

            # Helper to compute average score of a query's top-k results
            def avg_score(entry):
                scores = [
                    float(r.get("score", 0.0))
                    for r in entry.get("top_k_results", [])
                    if isinstance(r.get("score"), (int, float, str))
                ]
                return sum(scores) / len(scores) if scores else 0.0

            # Calculate deltas for shared queries
            diffs = []
            for query in sorted(shared_queries):
                s1 = avg_score(q1[query])
                s2 = avg_score(q2[query])
                delta = s2 - s1
                diffs.append((query, round(s1, 4), round(s2, 4), round(delta, 4)))

            # Sort by absolute delta
            diffs.sort(key=lambda x: abs(x[3]), reverse=True)

            # Build and send report
            report_lines = [
                f"Comparing:",
                f"- {file1} ({len(eval1)} queries)",
                f"- {file2} ({len(eval2)} queries)",
                "",
                f"Shared Queries: {len(shared_queries)}",
                f"Only in {file1}: {len(only_in_1)}",
                f"Only in {file2}: {len(only_in_2)}",
                "",
                "Score Differences on Shared Queries (Top 5 by delta):",
            ]
            for query, s1, s2, delta in diffs[:5]:
                report_lines.append(
                    f'- "{query[:40]}...": {s1} → {s2} (Δ {delta:+.4f})'
                )

            await cl.Message(
                author="Evaluation Bot", content="\n".join(report_lines)
            ).send()

        except Exception as e:
            await cl.Message(
                author="Evaluation Bot", content=f"Failed to compare logs: {e}"
            ).send()

        return True  # Trigger matched and handled

    return False  # Not a match


# Evaluation Trigger: Freeze and Export Session Evaluation
async def handle_eval_freeze_trigger(content: str) -> bool:
    """
    Handles the evaluation freeze trigger (!freeze_eval) — exports logs and summaries.
    Returns True if this trigger was matched and handled.
    """
    if content.strip().lower() != EVAL_TRIGGER_KEYWORD:
        return False

    eval_data = cl.user_session.get("eval_data", [])

    # Use session ID as a subfolder
    session_id = cl.user_session.get("eval_start_time", datetime.now().isoformat())
    session_slug = session_id.replace(":", "-").replace(".", "_")
    eval_dir = os.path.join(EVAL_FOLDER, f"session_{session_slug}")
    os.makedirs(eval_dir, exist_ok=True)

    # Timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_format = os.getenv("EVAL_OUTPUT_FORMAT", "json").lower()

    # Save evaluation log as JSON
    if eval_format in ["json", "both"]:
        json_path = os.path.join(eval_dir, f"eval_log_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)

    # Save evaluation log as CSV
    if eval_format in ["csv", "both"]:
        csv_path = os.path.join(eval_dir, f"eval_log_{timestamp}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "query",
                    "timestamp",
                    "chunk_id",
                    "score",
                    "model",
                    "source_file",
                    "token_count",
                    "matched_manual",   
                ]
            )
            for entry in eval_data:
                for chunk in entry.get("top_k_results", []):
                    writer.writerow(
                        [
                            entry["query"],
                            entry["timestamp"],
                            chunk.get("id", ""),
                            chunk.get("score", ""),
                            chunk.get("model", ""),
                            chunk.get("source_file", ""),
                            chunk.get("token_count", ""),
                            entry.get("matched_manual", False),
                        ]
                    )

    # Compute evaluation summary
    summary = summarize_eval_data(eval_data)

    # Optional Markdown export
    if os.getenv("EVAL_EXPORT_MARKDOWN", "false").lower() == "true":
        base_name = os.getenv("EVAL_MARKDOWN_FILENAME", "summary")
        markdown_path = os.path.join(eval_dir, f"{base_name}_{timestamp}.md")
        with open(markdown_path, "w", encoding="utf-8") as md:
            md.write(f"# Evaluation Summary ({timestamp})\n\n")
            for line in format_eval_summary(summary):
                md.write(f"{line}\n")

        await cl.Message(
            author="Evaluation Bot",
            content=f"Markdown summary exported to `{markdown_path}`",
        ).send()

    # Send summary to Chainlit
    summary_lines = format_eval_summary(summary)

    await cl.Message(author="Evaluation Bot", content="\n".join(summary_lines)).send()

    # Export summary as JSON or CSV
    if eval_format in ["json", "both"]:
        summary_path = os.path.join(eval_dir, f"eval_summary_{timestamp}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    if eval_format in ["csv", "both"]:
        summary_csv_path = os.path.join(eval_dir, f"eval_summary_{timestamp}.csv")
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for line in format_eval_summary(summary):
                writer.writerow([line])

    # Create ZIP (after all files are saved!)
    if EVAL_ZIP_EXPORT:
        zip_path = os.path.join(EVAL_FOLDER, f"{session_slug}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(eval_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, start=eval_dir)
                    zipf.write(file_path, arcname=arcname)

        # Offer ZIP for download (if enabled)
        valid_displays = {"inline", "side", "page"}
        if os.getenv("EVAL_OFFER_DOWNLOAD", "true").lower() == "true":
            await cl.Message(
                author="Evaluation Bot",
                content="Your evaluation summary is ready for download. ",
                elements=[
                    cl.File(
                        name=os.path.basename(zip_path),
                        path=zip_path,
                        display="inline",
                    )
                ],
            ).send()

        # Notify completion
        await cl.Message(
            author="Evaluation Bot",
            content=(
                f"Evaluation log saved to `{eval_dir}` with {len(eval_data)} entries.\n"
                f"Zipped archive: `{zip_path}`"
            ),
        ).send()
    else:
        await cl.Message(
            author="Evaluation Bot",
            content=f"Evaluation log saved to `{eval_dir}` with {len(eval_data)} entries.",
        ).send()

    return True  # Trigger handled


# Weather Logic
async def handle_weather_query(query: str):
    locations = []
    query_payload = {"inputs": query}
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    api_url = "https://api-inference.huggingface.co/models/dslim/bert-base-NER"

    async with cl.Step(name="Detecting Location", type="run") as step_locate:
        print("\033[94m[INFO]\033[0m Detecting locations using HuggingFace NER...")

        response = requests.post(api_url, json=query_payload, headers=headers)

        if response.status_code == 200:
            response_json = response.json()
            for entity in response_json:
                if entity.get("entity_group") == "LOC":
                    locations.append(entity["word"])
            step_locate.output = f"Found {len(locations)} location(s)."
            print(f"\033[92m[SUCCESS]\033[0m Locations detected: {locations}")
        else:
            step_locate.output = "NER API failed."
            print(
                f"\033[91m[ERROR]\033[0m HuggingFace NER failed with status code {response.status_code}"
            )
            await cl.Message(content="Failed to detect location.").send()
            return

        if not locations:
            step_locate.output = "No location found."
            print("\033[93m[WARNING]\033[0m No location entities found in user query.")
            await cl.Message(content="No locations found in your query.").send()
            return

    tense = detect_tense_simple(query)
    print(f"\033[90m └── Detected tense:\033[0m {tense or 'unknown'}")

    msg = cl.Message(content="", author="OpenWeather Bot")

    for location in locations:
        async with cl.Step(
            name=f"Fetching Weather: {location}", type="run"
        ) as step_weather:
            try:
                weather_url = (
                    f"https://api.openweathermap.org/data/2.5/forecast?q={location}&units=metric&appid={OPENWEATHER_TOKEN}"
                    if tense == "FUT"
                    else f"https://api.openweathermap.org/data/2.5/weather?q={location}&units=metric&appid={OPENWEATHER_TOKEN}"
                )

                print(f"\033[94m[INFO]\033[0m Fetching weather data for {location}...")

                weather_response = requests.get(weather_url)
                if weather_response.status_code != 200:
                    step_weather.output = "API Error"
                    print(
                        f"\033[91m[ERROR]\033[0m Failed to retrieve weather for {location} (code {weather_response.status_code})"
                    )
                    await msg.stream_token(
                        f"Failed to retrieve weather data for {location}.\n"
                    )
                    continue

                data = weather_response.json()

                # Prepare raw text summary
                if tense == "FUT":
                    forecast = data["list"][:3]
                    raw_weather = "\n".join(
                        f"{entry['dt_txt']}: {entry['main']['temp']}°C, {entry['weather'][0]['description']}"
                        for entry in forecast
                    )
                else:
                    raw_weather = (
                        f"Temperature: {data['main']['temp']}°C\n"
                        f"Condition: {data['weather'][0]['description']}\n"
                        f"Humidity: {data['main']['humidity']}%\n"
                        f"Wind: {data['wind']['speed']} m/s"
                    )

                # Create prompt for weather synthesis
                weather_prompt = f"""
You are MarinEnGPT, a professional assistant specializing in marine and coastal weather reporting.

The user asked: "{query}"

Below is the weather data for location: {location}
Timeframe: {"Future forecast" if tense == "FUT" else "Current conditions"}

Weather Data:
{raw_weather}

Based on this data, generate a helpful summary with:
- Key weather conditions
- Any precautions or recommendations
- Professional and concise tone

Answer:"""

                # Call LLM to summarize
                llm_response = Settings.llm.complete(weather_prompt)
                response_text = llm_response.text.strip()

                await msg.stream_token(response_text + "\n")
                step_weather.output = "Weather summarized."
                print(f"\033[92m[SUCCESS]\033[0m LLM weather synthesis complete.")

            except Exception as e:
                step_weather.output = "Exception"
                print(
                    f"\033[91m[ERROR]\033[0m Exception fetching weather for {location}: {e}"
                )
                await msg.stream_token(f"Error fetching weather for {location}: {e}\n")

    await msg.send()


# This will do the false-positives for weather instead of semantic search
def is_weather_query(text: str) -> bool:
    threshold = int(os.getenv("WEATHER_KEYWORD_THRESHOLD", 2))
    text = text.lower()
    keyword_hits = [kw for kw in weather_keywords if kw in text]

    print(
        f"\033[90m[DEBUG]\033[0m Detected {len(keyword_hits)} weather keywords: {keyword_hits}"
    )

    return len(keyword_hits) >= threshold


async def handle_weather_trigger(content: str) -> bool:
    """
    If the content is a weather query, call `handle_weather_query()` directly.
    """
    if is_weather_query(content):
        await handle_weather_query(content)  # Call the original function
        return True
    return False


# Semantic Querying via ChromaDB Index
async def handle_semantic_query(
    content: str, message: cl.Message, enable_context: bool, chat_history: list
) -> bool:
    response_text = ""
    retrieval_eval = None
    response_eval = None  # Placeholder for quality evaluation result
    response_evaluator = QueryResponseEvaluator(
        llm=Settings.llm
    )  # Instantiate the evaluator

    if index is None:
        await cl.Message(
            content="No embeddings are currently loaded. Please try again after loading the collection."
        ).send()
        return

    # Get chat history context toggle and state
    if enable_context:
        chat_history.append({"role": "assistant", "content": response_text})
        cl.user_session.set("chat_history", chat_history)

    # Step 1: Retrieving relevant chunks
    async with cl.Step(name="Retrieving Chunks", type="run") as step_retrieve:
        top_k = int(os.getenv("TOP_K_RETRIEVAL_RESULTS", 5))
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(content)

        # Patch .text field if missing
        for node in nodes:
            # Check if node.text is empty or looks like a fallback
            if not node.text.strip() or node.text.startswith("# Response for"):
                fallback_text = node.metadata.get("document") or node.metadata.get(
                    "documents"
                )
                if fallback_text:
                    node.text = fallback_text  # Patch in the real chunk content

        # Now filter valid nodes
        valid_nodes = [
            n
            for n in nodes
            if hasattr(n, "text") and isinstance(n.text, str) and n.text.strip()
        ]

        # Detect whether retrieved chunks match the QA_SOURCE_FILE_FILTER
        retrieved_manuals = {
            node.metadata.get("source_file", "").lower()
            for node in valid_nodes
        }
        qa_filter = QA_SOURCE_FILE_FILTER.lower() if QA_SOURCE_FILE_FILTER else None

        manual_match = any(qa_filter in src for src in retrieved_manuals) if qa_filter else True

        # Evaluate retrieval after filtering valid
        retriever_evaluator = RetrieverEvaluator.from_metric_names(
            ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"],
            retriever=retriever,
        )
        # Retrieve QA dataset lookup
        qa_lookup = cl.user_session.get("qa_lookup")

        # STEP 1: QA dataset generation (top-level)
        # Lazy-load QA dataset on first semantic query if needed
        if qa_lookup is None:
            if not ENABLE_QA_DATASET_GENERATION:
                print("\033[94m[INFO]\033[0m Synthetic QA dataset generation is disabled. Skipping retrieval evaluation.")
                cl.user_session.set("qa_lookup", {})  # Set to empty to avoid re-entering
            else:
                async with cl.Step(name="Generate QA Dataset", type="run") as step_qa_gen:
                    print(f"\033[90m[DEBUG]\033[0m Generating synthetic QA dataset...")
                    print(
                        f"\033[90m[DEBUG]\033[0m Generating synthetic QA dataset for evaluation "
                        f"(questions_per_chunk={QA_NUM_QUESTIONS}, "
                        f"embedding_mode={QA_EMBEDDING_MODE_FILTER}, "
                        f"source_file={QA_SOURCE_FILE_FILTER})..."
                    )
                    dataset = generate_qa_dataset(
                        num_questions=QA_NUM_QUESTIONS,
                        embedding_mode_filter=QA_EMBEDDING_MODE_FILTER,
                        source_file_filter=QA_SOURCE_FILE_FILTER,
                    )
                    # dd(dataset[0:2])
                    qa_lookup = {}

                    # Apparently, RagDatasetGenerator does not have key-value pair for reference node_ids so more manual labour for me..smfh
                    # Construct mapping: question → list of expected node IDs
                    for i, example in enumerate(dataset[:10]):  # limit to 10 for debug
                        print(f"\033[90m[DEBUG]\033[0m Example {i} type: {type(example)}")
                        print(f"\033[90m[DEBUG]\033[0m Query: {example.query}")
                        print(
                            f"\033[90m[DEBUG]\033[0m Contexts: {example.reference_contexts}"
                        )

                        if not example.query or not example.reference_contexts:
                            continue

                        expected_ids = []
                        for context in example.reference_contexts:
                            match = re.search(r"# Response for ([^\n]+)", context)
                            if match:
                                expected_ids.append(match.group(1).strip())

                        if expected_ids:
                            qa_lookup[example.query] = expected_ids
                    # [DEBUG] Preview generated QA lookup entries to verify mapping accuracy
                    print("\033[90m[DEBUG]\033[0m QA Lookup table initialized with",
                        len(qa_lookup), "entries")
                    for i, (q, ids) in enumerate(list(qa_lookup.items())[:3]):
                        print(f" ├─ Q{i+1}: {q[:80]}...")
                        print(f" └─ Expected IDs: {ids}")

                    cl.user_session.set("qa_lookup", qa_lookup)

                    step_qa_gen.output = f"{len(qa_lookup)} QA pairs created." 

                print("\033[94m[INFO]\033[0m Retrieval Evaluation Setup:")
                print(" - Evaluation metrics (hit rate, MRR, precision, recall, etc.) are based on exact or fuzzy matches to the expected chunk IDs.")
                print(" - These expected IDs are derived from a synthetic QA dataset using the MF-194 manual.")
                print(" - MF-194 is chosen because it is small and avoids OpenAI rate limits.")
                print(" - Fuzzy matching is enabled to tolerate paraphrased queries and provide more robust evaluations.")
                print(f" - Fuzzy match threshold: {QA_MATCH_THRESHOLD:.2f}\n")

        async with cl.Step(name="Match QA Query", type="run") as step_match:
            matched_query, expected_ids = find_closest_qa_query(content, qa_lookup)

            if matched_query:
                step_match.output = "QA match found."
            else:
                step_match.output = "No QA match (threshold not met)."

        if expected_ids:
            print(f"[DEBUG] type(expected_ids): {type(expected_ids)}")
            print(f"[DEBUG] expected_ids repr: {repr(expected_ids)}")
            print(f"[DEBUG] expected_ids truthiness: {'yes' if expected_ids else 'no'}")

            eval_nodes = []

            print("[DEBUG] Raw node IDs before normalization:")
            for node in valid_nodes:
                # Extract base node and print original node_id
                base_node = node.node if isinstance(node, NodeWithScore) else node
                print(f" - {base_node.node_id}")

                # Normalize: strip "_chunk_#" suffix to match expected full-page IDs
                norm_id = base_node.node_id.split("_chunk_")[0] if "_chunk_" in base_node.node_id else base_node.node_id

                # Create a shallow copy of the node and override node_id
                base_node_copy = base_node.copy()
                base_node_copy.node_id = norm_id  # Needed for accurate comparison inside evaluator

                # Wrap in NodeWithScore using normalized ID and appropriate score
                eval_nodes.append(
                    NodeWithScore(
                        node=base_node_copy,
                        score=node.score if isinstance(node, NodeWithScore) else 1.0,
                        node_id=norm_id,
                    )
                )
            # DEBUG: Confirm evaluator sees the correct node IDs
            print("[DEBUG] Effective node IDs seen by evaluator:")
            for e in eval_nodes:
                print(f" - Wrapper node_id: {e.node_id} | Inner node.node_id: {e.node.node_id}")

            # Print normalized IDs to be passed to evaluator
            print("[DEBUG] Normalized node IDs passed to evaluator:")
            for node in eval_nodes:
                print(f" - {repr(node.node_id)}")

            # Print expected IDs
            print("[DEBUG] Expected IDs for QA match:")
            for eid in expected_ids:
                print(f" - {repr(eid)}")

            # Compare each normalized retrieved node ID to each expected ID
            print("[DEBUG] Comparing normalized node IDs to expected IDs:")
            for node in eval_nodes:
                for eid in expected_ids:
                    print(f"  - {node.node_id.strip()} == {eid.strip()} ? {node.node_id.strip() == eid.strip()}")

            # Show intersection set for final confirmation
            matched_ids = {n.node_id.strip() for n in eval_nodes}
            expected_set = {e.strip() for e in expected_ids}
            intersection = matched_ids & expected_set
            print(f"[DEBUG] Matched IDs intersection: {intersection}")

            async with cl.Step(name="Evaluate Retrieval", type="run") as step_eval:
                # Run LlamaIndex retrieval evaluator
                retrieval_result = retriever_evaluator.evaluate(
                    query=content,
                    retrieved_nodes=eval_nodes,
                    expected_ids=expected_ids,
                )

                retrieval_eval = retrieval_result if isinstance(retrieval_result, dict) else {}

                if retrieval_eval:
                    step_eval.output = "Evaluator returned metrics."
                else:
                    step_eval.output = "Evaluator returned nothing."
                print(f"[EVAL][Retrieval Metrics] Matched to QA query: {matched_query}")
                print(f"[EVAL][Retrieval Metrics] {retrieval_eval}")

            if not retrieval_eval:
                print("[WARNING] Evaluator returned empty result. Falling back to manual check.")

                async with cl.Step(name="Fallback Evaluation", type="run") as step_fallback:
                    # Manual fallback evaluation for sanity check
                    matched_ids = {n.node_id.strip() for n in eval_nodes}
                    expected_set = {e.strip() for e in expected_ids}

                    manual_hit = bool(matched_ids & expected_set)
                    print(f"[DEBUG] Manual Hit Detected: {manual_hit}")

                    retrieval_eval = {
                        "hit_rate": 1.0 if manual_hit else 0.0,
                        "mrr": 1.0 if manual_hit else 0.0,
                        "precision": 1.0 if manual_hit else 0.0,
                        "recall": 1.0 if manual_hit else 0.0,
                    } if manual_hit else {}

                    step_fallback.output = (
                        "Manual match successful." if manual_hit else "Manual match failed."
                    )

        else:
            retrieval_eval = {}  # No match — skip evaluation

        # Warn if current query is not from the QA dataset's filtered manual
        if qa_filter and not manual_match:
            print(
                f"\033[93m[WARNING]\033[0m Query appears unrelated to '{QA_SOURCE_FILE_FILTER}'. "
                "Retrieval metrics may be inaccurate or meaningless.\n"
            )

        if not valid_nodes:
            step_retrieve.output = "No content retrieved."
            await cl.Message(
                content="No valid content was found in the vector store to answer your question."
            ).send()
            return

        step_retrieve.output = f"{len(valid_nodes)} chunks found."

    # Log top-k results
    log_top_k_results(valid_nodes, k=TOP_K_RESULTS)

    # Step 2: Generating Response with conditional features of Decoupled Retrieval & Synthesis
    async with cl.Step(name="Generating Response", type="run") as step_generate:
        msg = cl.Message(author="MarinEnGPT", content="")
        await msg.send()

        print("\n\033[96m" + "=" * 60 + "\033[0m")
        print("\033[1;94m[Query Input]\033[0m")
        print(f"{content}\n")

        if ENABLE_DECOUPLED_SYNTHESIS:
            print("[INFO] Decoupled Retrieval & Synthesis is ENABLED")
            synthesis_units = group_chunks_for_synthesis(
                valid_nodes,
                group_field=CHUNK_GROUP_FIELD,
                min_size=CHUNK_SYNTHESIS_MIN,
                max_size=CHUNK_SYNTHESIS_MAX,
                ordered=CHUNK_SYNTHESIS_ORDERED,
            )

            if not synthesis_units:
                step_generate.output = "No valid synthesis groups."

                # Load last-used CLI collection info for helpful comparison
                cli_config = load_last_chroma_collection_config()
                last_used = cli_config.get("chromadb_collection_name")

                print(
                    "\033[91m[ERROR]\033[0m No valid chunk groups found for synthesis."
                )
                print(
                    "\033[90m └─ Make sure you're querying a collection with properly chunked embeddings.\033[0m"
                )
                if CHROMA_COLLECTION != last_used:
                    print(
                        f"\033[33m[HINT]\033[0m Last CLI-selected collection was: {last_used}"
                    )

                await cl.Message(
                    content=(
                        "No valid content groups were found to generate a synthesized answer.\n"
                        "Please check if your current collection includes chunked embeddings."
                    )
                ).send()
                return

            # Optionally include short-term chat history in synthesis prompt
            context_prefix = ""
            if enable_context and chat_history:
                recent_turns = chat_history[-6:]  # Limit to 3 exchanges
                formatted = [
                    f"{turn['role'].capitalize()}: {turn['content'].strip()}"
                    for turn in recent_turns
                ]
                context_prefix = "\n".join(formatted) + "\n\n"

            combined_prompt = "\n\n".join(text for _, text in synthesis_units)

            synthesis_prompt = f"""You are MarinEnGPT, an expert AI assistant in marine engineering systems, especially service and maintenance procedures for shipboard equipment.

            {context_prefix}Use the following technical context entries — which you have already internalized — to answer the user's question clearly, concisely, and accurately.

            Do not mention documents, excerpts, or sources. Only answer based on the given context. If the context lacks the information, say so directly.

            User Question:
            \"{content}\"

            Context:
            {combined_prompt}

            Answer:"""

            llm_response = Settings.llm.complete(synthesis_prompt)
            response_text = llm_response.text

            # Evaluate the response
            # Wrap response_text so it has a .response attribute for evaluation
            response_eval = response_evaluator.evaluate_response(
                query=content,
                response=LLMResponseWrapper(response_text, source_nodes=valid_nodes),
            )

            # Session-Based Evaluation Logging: After response evaluation
            if (
                ENABLE_SESSION_EVAL
                and not is_weather_query(content)
                and not message.elements
            ):
                eval_data = cl.user_session.get("eval_data", [])
                eval_entry = {
                    "query": content,
                    "timestamp": datetime.now().isoformat(),
                    "session_id": cl.user_session.get("eval_start_time", "unknown"),
                    "top_k_results": [
                        {
                            "id": node.id_,
                            "score": getattr(node, "score", "N/A"),
                            "model": node.metadata.get("model", "N/A"),
                            "source_file": node.metadata.get("source_file", "N/A"),
                            "token_count": node.metadata.get("token_count", "N/A"),
                        }
                        for node in valid_nodes[:TOP_K_RESULTS]
                    ],
                    "retriever_metrics": retrieval_eval,
                    "response_quality": response_eval.dict() if response_eval else None,
                    "matched_manual": manual_match,
                }
                eval_data.append(eval_entry)

                # Enforce max log size
                if (
                    EVAL_MAX_LOGGED_QUERIES > 0
                    and len(eval_data) > EVAL_MAX_LOGGED_QUERIES
                ):
                    eval_data = eval_data[-EVAL_MAX_LOGGED_QUERIES:]

                cl.user_session.set("eval_data", eval_data)

            if response_eval:
                score = response_eval.score
                passing = response_eval.passing
                feedback = response_eval.feedback or "N/A"

                color_pass = "\033[92m" if passing else "\033[91m"  # Green or Red
                reset = "\033[0m"

                print(
                    f"\033[93m[EVAL]\033[0m Response Quality → "
                    f"Score: \033[96m{score}\033[0m, "
                    f"Passed: {color_pass}{passing}{reset}, "
                    f"Feedback: \033[90m{feedback}{reset}"
                )

            # Stream the response slowly for natural effect
            for token in response_text.split():
                await msg.stream_token(token + " ")
                await asyncio.sleep(0.02)

            msg.content = response_text
            await msg.update()

            if enable_context:
                chat_history.append({"role": "assistant", "content": response_text})
                cl.user_session.set("chat_history", chat_history)

            step_generate.output = f"{len(synthesis_units)} group(s) synthesized."

            print("\033[1;95m[Synthesized Output]\033[0m")
            print(response_text)
            print("\033[96m" + "=" * 60 + "\033[0m\n")
        else:
            print("[INFO] Decoupled Retrieval & Synthesis is DISABLED")

            streaming_engine = index.as_query_engine(streaming=True)
            response = streaming_engine.query(content)

            response_text = ""
            for token in response.response_gen:
                response_text += str(token)
                await msg.stream_token(token)
                await asyncio.sleep(
                    0.01
                )  # Slow down streaming rate - It should stop Engineio throwing 'Too many packets in payload' error

            msg.content = response_text
            await msg.update()

            step_generate.output = "Response delivered (streamed)."

            print("\033[1;95m[Streamed Response Text]\033[0m")
            print(response_text)
            print("\033[96m" + "=" * 60 + "\033[0m\n")

        if not response_text.strip():
            step_generate.output = "Empty response."
            await cl.Message(
                content="No relevant information was found in the vector database."
            ).send()
        else:
            step_generate.output = "Response delivered."

    # Step 3: Displaying Top 3 References
    show_references = os.getenv("SHOW_REFERENCES", "True").lower() == "true"
    if valid_nodes:
        top_references = valid_nodes[:3]
        references_payload = [
            {
                "id": node.id_,
                "source_file": node.metadata.get("source_file", "Unknown source"),
                "text": node.text,
            }
            for node in top_references
        ]

        # Define actions for user to choose how to view references
        if show_references:
            actions = [
                cl.Action(
                    name="view_as_image",
                    label="View as Image",
                    payload={"references": references_payload},
                ),
                cl.Action(
                    name="view_as_markdown",
                    label="View as Markdown",
                    payload={"references": references_payload},
                ),
            ]

            # Send a message with action buttons for user to choose
            await cl.Message(
                content="How would you like to view the references?",
                author="Reference Bot",
                actions=actions,
            ).send()


@cl.on_message
async def main(message: cl.Message):
    content = message.content

    # Get chat history context toggle and state
    enable_context = cl.user_session.get("enable_chat_history", False)
    chat_history = cl.user_session.get("chat_history", [])

    print(f"\033[90m[DEBUG]\033[0m Short-term context enabled: {enable_context}")

    # Append user query to history if enabled
    if enable_context:
        print(
            f"\033[90m[DEBUG]\033[0m Chat history turns in memory (Q + A): {len(chat_history)}"
        )
        chat_history.append({"role": "user", "content": content})
        cl.user_session.set("chat_history", chat_history)

    # Handle Evaluation Query Triggers
    if await handle_eval_list_trigger(content):
        return

    if await handle_eval_replay_trigger(content):
        return

    if await handle_eval_compare_trigger(content):
        return

    if await handle_eval_freeze_trigger(content):
        return

    # Handle Weather Query Trigger
    if await handle_weather_trigger(content):
        return

    # Handle File Uploads (PDF → OCR/Image Extraction)
    if message.elements:
        await cl.Message(
            content="The uploaded files are now being processed...", author="MarinEnGPT"
        ).send()
        pdf_paths = {
            file.path for file in message.elements if "application/pdf" in file.mime
        }

        for pdf_path in pdf_paths:
            ocr_generator(pdf_path)
            extract_images_from_pdf(
                pdf_path, output_folder="./.files", preprocess=False
            )

        await cl.Message(
            content="Files have been processed. Querying them will be supported soon."
        ).send()
        return

    # Handle Semantic Querying
    if await handle_semantic_query(content, message, enable_context, chat_history):
        return
