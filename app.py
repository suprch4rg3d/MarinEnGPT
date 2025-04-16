import os
import requests
from typing import Optional, Dict
import chainlit as cl

from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.query_engine import SimpleMultiModalQueryEngine

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
            "chromadb_persistence_path": config.get("chromadb_persistence_path", "./data/chromadb"),
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
            print(f" └─ Content Snippet (metadata.document): {ANSI_GRAY}[EMPTY]{ANSI_RESET}\n")
        else:
            doc_snippet = shorten(doc_fallback or "", width=100, placeholder="...")
            print(f' └─ Content Snippet (metadata.document): {ANSI_GRAY}"{doc_snippet}"{ANSI_RESET}\n')


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

# This will do the false-positives for weather instead of semantic search
def is_weather_query(text: str) -> bool:
    threshold = int(os.getenv("WEATHER_KEYWORD_THRESHOLD", 2))
    text = text.lower()
    keyword_hits = [kw for kw in weather_keywords if kw in text]

    print(
        f"\033[90m[DEBUG]\033[0m Detected {len(keyword_hits)} weather keywords: {keyword_hits}"
    )

    return len(keyword_hits) >= threshold

@cl.on_chat_start
async def start():
    print(f"\033[90m[DEBUG]\033[0m Top-K retrieval log threshold: {TOP_K_RESULTS}")
    
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

    elements = []

    for idx, ref in enumerate(references, 1):
        node_id = ref.get("id")
        base_id = node_id.rsplit("_page_", 1)[0]
        image_path = os.path.join(IMAGE_BASE_PATH, base_id, f"{node_id}.png")

        if os.path.exists(image_path):
            image_element = cl.Image(
                path=image_path,
                name=f"Reference Image {idx}",
                display="inline",
                size="medium",
            )
            elements.append(image_element)
        else:
            await cl.Message(
                content=f"Warning: Image for reference {idx} not found at {image_path}."
            ).send()

    if elements:
        await cl.Message(
            content="Here are the references as images:", elements=elements
        ).send()
    else:
        await cl.Message(
            content="No images available for the selected references."
        ).send()


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

    # Handle Weather Queries
    if is_weather_query(content):
        await handle_weather_query(content)
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

    # Semantic Query via ChromaDB Index
    if index is None:
        await cl.Message(
            content="No embeddings are currently loaded. Please try again after loading the collection."
        ).send()
        return

    # Step 1: Retrieving relevant chunks
    async with cl.Step(name="Retrieving Chunks", type="run") as step_retrieve:
        top_k = int(os.getenv("TOP_K_RETRIEVAL_RESULTS", 5))
        retriever = index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(content)

        # Patch .text field if missing
        for node in nodes:
            # Check if node.text is empty or looks like a fallback
            if not node.text.strip() or node.text.startswith("# Response for"):
                fallback_text = node.metadata.get("document") or node.metadata.get("documents")
                if fallback_text:
                    node.text = fallback_text  # Patch in the real chunk content

        # Now filter valid nodes
        valid_nodes = [
            n for n in nodes if hasattr(n, "text") and isinstance(n.text, str) and n.text.strip()
        ]

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

                print("\033[91m[ERROR]\033[0m No valid chunk groups found for synthesis.")
                print("\033[90m └─ Make sure you're querying a collection with properly chunked embeddings.\033[0m")
                if CHROMA_COLLECTION != last_used:
                    print(f"\033[33m[HINT]\033[0m Last CLI-selected collection was: {last_used}")

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
                cl.Action(
                    name="skip_references",
                    label="Skip References",
                    payload={},
                ),
            ]

            # Send a message with action buttons for user to choose
            await cl.Message(
                content="How would you like to view the references?",
                author="Reference Bot",
                actions=actions,
            ).send()


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
