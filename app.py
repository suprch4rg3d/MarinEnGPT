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
from collections import defaultdict
import re

# API KEYS
ENABLE_USER_ENV = os.getenv("ENABLE_USER_ENV", "false").lower() == "true"
OPENAI_TOKEN: Optional[str] = os.getenv("OPENAI_API_KEY")
HF_TOKEN: Optional[str] = os.getenv("HUGGING_FACE_TOKEN")
OPENWEATHER_TOKEN: Optional[str] = os.getenv("OPENWEATHER_API_KEY")


# Set default LLM and embedding model
llm = OpenAI(model="gpt-4o", temperature=0.0)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")

Settings.llm = llm
Settings.embed_model = embed_model

# Path and collection for ChromaDB
CHROMA_PATH = "./data/chromadb"
CHROMA_COLLECTION = "MarineEngineeringManuals"


# Connect to ChromaDB and load pre-generated index
def load_chromadb_index():
    print("\033[94m[INFO]\033[0m Connecting to ChromaDB...")

    client = PersistentClient(
        path=CHROMA_PATH,
        settings=ChromaSettings(anonymized_telemetry=False),  # Disable telemetry
    )

    collections = client.list_collections()
    if CHROMA_COLLECTION not in collections:
        raise ValueError(
            f"\033[91m[ERROR]\033[0m Collection '{CHROMA_COLLECTION}' not found in {CHROMA_PATH}."
        )

    chroma_collection = client.get_collection(name=CHROMA_COLLECTION)
    doc_ids = chroma_collection.get()["ids"]
    doc_count = len(doc_ids)

    print(f"\033[92m[SUCCESS]\033[0m Connected to collection: '{CHROMA_COLLECTION}'")
    print(f"\033[90m ├── Documents loaded:\033[0m {doc_count}")

    # --- Customizable Preview Settings ---
    show_group_preview = os.getenv("GROUP_PREVIEW_ENABLED", "true").lower() == "true"
    group_preview_limit = int(os.getenv("GROUP_PREVIEW_LIMIT", 5))

    if show_group_preview:
        # Group based on ID prefix (e.g. 'MF-200_page_3' -> 'MF-200')
        page_pattern = re.compile(r"^(.*)_page_\d+$")
        grouped = defaultdict(list)
        for doc_id in doc_ids:
            match = page_pattern.match(doc_id)
            base_id = match.group(1) if match else doc_id
            grouped[base_id].append(doc_id)

        print(f"\033[90m ├── Document Groups:\033[0m {len(grouped)}")
        for group, entries in sorted(grouped.items())[:group_preview_limit]:
            print(f"     ├─ {group}  ({len(entries)} pages)")
        if len(grouped) > group_preview_limit:
            print(f"     └─ ...and {len(grouped) - group_preview_limit} more groups.")

    print(f"\033[90m └── Vector DB path:\033[0m {CHROMA_PATH}\n")

    if doc_count == 0:
        print(
            "\033[93m[WARNING]\033[0m Collection is empty. Semantic queries will be skipped.\n"
        )
        return None  # New: Return None for empty collection

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load index directly from Chroma collection
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


@cl.on_chat_start
async def start():
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

    # Handle Weather Queries
    if any(keyword in content for keyword in weather_keywords):
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
        retriever = index.as_retriever()
        nodes = retriever.retrieve(content)
        valid_nodes = [
            n
            for n in nodes
            if hasattr(n, "text") and isinstance(n.text, str) and n.text.strip()
        ]

        if not valid_nodes:
            step_retrieve.output = "No content retrieved."
            await cl.Message(
                content="No valid content was found in the vector store to answer your question."
            ).send()
            return

        step_retrieve.output = f"{len(valid_nodes)} chunks found."

    # Log top-k results
    log_top_k_results(valid_nodes)

    # Step 2: Generating Response
    async with cl.Step(name="Generating Response", type="run") as step_generate:
        streaming_engine = index.as_query_engine(streaming=True)
        msg = cl.Message(author="MarinEnGPT", content="")
        await msg.send()

        # Run query and stream response
        response = streaming_engine.query(content)

        print("\n\033[96m" + "=" * 60 + "\033[0m")
        print("\033[1;94m[Query Input]\033[0m")
        print(f"{content}\n")

        response_text = ""
        for token in response.response_gen:
            response_text += str(token)
            await msg.stream_token(token)

        print("\033[1;95m[Streamed Response Text]\033[0m")
        print(response_text)
        print("\033[96m" + "=" * 60 + "\033[0m\n")

        msg.content = response_text
        await msg.update()

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

        # Use HuggingFace to extract location entities
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
        async with cl.Step(name=f"Fetching Weather: {location}", type="run") as step_weather:
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

                async with cl.Step(name=f"Streaming Weather Info: {location}", type="run") as step_stream:
                    if tense == "FUT":
                        forecast = data["list"][:3]
                        await msg.stream_token(f"**Forecast for {location}:**\n\n")
                        for entry in forecast:
                            dt_txt = entry["dt_txt"]
                            temp = entry["main"]["temp"]
                            desc = entry["weather"][0]["description"]
                            await msg.stream_token(f"- {dt_txt}: {temp}°C, {desc}\n")
                    else:
                        temp = data["main"]["temp"]
                        desc = data["weather"][0]["description"]
                        humidity = data["main"]["humidity"]
                        wind = data["wind"]["speed"]
                        await msg.stream_token(
                            f"**Current weather in {location}:**\n"
                            f"- Temperature: {temp}°C\n"
                            f"- Condition: {desc}\n"
                            f"- Humidity: {humidity}%\n"
                            f"- Wind Speed: {wind} m/s\n\n"
                        )
                        step_stream.output = "Weather sent."
                    print(f"\033[92m[SUCCESS]\033[0m Weather data fetched for {location}")
                    step_weather.output = "Weather fetched."

            except Exception as e:
                step_weather.output = "Exception"
                print(
                    f"\033[91m[ERROR]\033[0m Exception fetching weather for {location}: {e}"
                )
                await msg.stream_token(f"Error fetching weather for {location}: {e}\n")

    await msg.send()
