import os
import json
import openai
import time
from datetime import datetime, timezone
import logging
import pyperclip
import requests
from helpers import *
from dotenv import load_dotenv
import collections
import tiktoken
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from math import ceil
from textwrap import wrap
from tqdm import tqdm
import re
from collections import defaultdict

# Global ChromaDB Client
chromadb_client = (
    None  # A single instance tracked globally - Use getter to access it!!!
)

# Persistence Configuration
PERSISTENCE_FILE = "embeddings_openai_persistence.json"


def load_config():
    """
    Loads the configuration file and returns stored values.
    If the file does not exist or is corrupted, it returns default values.

    Returns:
        dict: Dictionary containing stored values.
    """
    default_config = {
        "input_folder": "./data/markdown",
        "batch_input_file": "embeddings_batch_input.jsonl",
        "batch_output_file": "embeddings_batch_output.jsonl",
        "output_folder": "./data/vectors",
        "model": "text-embedding-3-small",
        "encoding_format": "float",
        "url": "/v1/embeddings",
        "dimensions": None,
        "use_dimensions": False,
        "chromadb_storage_mode": None,
        "chromadb_persistence_path": "./data/chromadb",
        "chromadb_collection_name": "MarineEngineeringManuals",
        "chromadb_distance_function": "cosine",
    }

    if not os.path.exists(PERSISTENCE_FILE):
        return default_config  # Return defaults if config file doesn't exist

    try:
        with open(PERSISTENCE_FILE, "r", encoding="utf-8") as file:
            saved_config = json.load(file)

        # Ensure all required keys are present
        for key in default_config:
            if key not in saved_config:
                saved_config[key] = default_config[key]

        return saved_config  # Load config from JSON file
    except (json.JSONDecodeError, IOError):
        return default_config  # If corrupted, return defaults


def save_config(config):
    """
    Saves the updated configuration to a JSON file.

    Args:
        config (dict): Dictionary containing updated paths.
    """
    try:
        with open(PERSISTENCE_FILE, "w", encoding="utf-8") as file:
            json.dump(config, file, indent=4)
        print("\nConfiguration saved successfully.")
    except IOError as e:
        print(f"\nError saving configuration: {e}")


# Logging Configuration
logging.basicConfig(
    filename="embeddings_openai.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)

# Load .env file if it exists
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")


def construct_request_object(
    custom_id,
    model,
    content,
    encoding_format,
    dimensions,
    use_dimensions,
    url="/v1/embeddings",
):
    """
    Constructs a request object for the OpenAI Embeddings API.

    Args:
        custom_id (str): Unique identifier for the request.
        model (str): Model to use for embeddings.
        content (str): Extracted text from Markdown file.
        encoding_format (str): Format of the returned embedding (e.g., "float").
        dimensions (int or None): Number of embedding dimensions (if specified).
        use_dimensions (bool): Whether to include custom dimensionality in the request.
        url (str): API endpoint URL (default: "/v1/embeddings").

    Returns:
        dict: Request object.
    """
    logging.info(f"Constructing embedding request for: {custom_id}")

    request_body = {
        "model": model,
        "input": content,
        "encoding_format": encoding_format,
    }

    # Include dimensions only if enabled
    if use_dimensions and dimensions:
        request_body["dimensions"] = dimensions

    request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": url,
        "body": request_body,
    }

    logging.debug(f"Constructed request object: {json.dumps(request, indent=2)}")
    return request


def create_batch_input_jsonl(
    input_folder,
    output_file,
    model,
    encoding_format,
    dimensions,
    use_dimensions,
    url="/v1/embeddings",
):
    """
    Creates a .jsonl file from Markdown files in the input folder, ensuring token limits are respected.

    Args:
        input_folder (str): Path to the folder containing Markdown files.
        output_file (str): Path to the output .jsonl file.
        model (str): Model to use for embeddings.
        encoding_format (str): Format of the returned embedding.
        dimensions (int or None): Number of embedding dimensions (if specified).
        use_dimensions (bool): Whether to include custom dimensionality.
        url (str): API endpoint URL (default: "/v1/embeddings").

    Raises:
        Exception: If there is an error during file creation.
    """
    logging.info("Creating batch input JSONL file for embeddings...")

    # Define model-specific max token limits
    model_max_tokens = {
        "text-embedding-3-small": 8191,
        "text-embedding-3-large": 8191,
        "text-embedding-ada-002": 8191,
    }

    max_tokens = model_max_tokens.get(
        model, 8191
    )  # Default to 8191 if model is unknown

    try:
        with open(output_file, "w") as outfile:
            for filename in os.listdir(input_folder):
                if filename.endswith(".md"):
                    # Read text from Markdown file
                    md_path = os.path.join(input_folder, filename)
                    with open(md_path, "r", encoding="utf-8") as md_file:
                        content = md_file.read().strip()

                    if not content:
                        logging.warning(f"Skipping empty Markdown file: {filename}")
                        continue

                    # Count tokens using dynamic encoding selection
                    num_tokens = count_tokens(content, model)

                    # Handle token limit
                    if num_tokens > max_tokens:
                        logging.warning(
                            f"File {filename} exceeds {max_tokens} tokens ({num_tokens} tokens). Truncating input..."
                        )
                        print(
                            f"\033[33mWarning: {filename} has {num_tokens} tokens (exceeds {max_tokens}). Truncating input...\033[0m"
                        )

                        # Dynamically determine the encoding for truncation
                        encoding = tiktoken.encoding_for_model(model)
                        tokens = encoding.encode(content)
                        content = encoding.decode(tokens[:max_tokens])
                        num_tokens = max_tokens  # Update count after truncation

                    # Construct request object
                    custom_id = os.path.splitext(filename)[0]  # Use filename as ID
                    request = construct_request_object(
                        custom_id=custom_id,
                        model=model,
                        content=content,
                        encoding_format=encoding_format,
                        dimensions=dimensions,
                        use_dimensions=use_dimensions,
                        url=url,
                    )

                    outfile.write(json.dumps(request) + "\n")

        logging.info(f"Batch input file created: {output_file}")
        print(f"Batch input file created: {output_file}")

    except Exception as e:
        logging.error(f"Error creating batch input file: {e}")
        raise


def upload_batch_file(batch_input_file):
    """
    Uploads a batch input file to OpenAI for processing.

    Args:
        batch_input_file (str): Path to the batch input .jsonl file.

    Returns:
        str: File ID assigned by OpenAI.

    Raises:
        Exception: If there is an error during upload.
    """
    try:
        response = openai.files.create(
            file=open(batch_input_file, "rb"), purpose="batch"
        )
        file_id = response.id
        logging.info(f"Batch input file uploaded. File ID: {file_id}")
        print(f"Batch input file uploaded. File ID: {file_id}")
        return file_id
    except Exception as e:
        logging.error(f"Error uploading batch input file: {e}")
        raise


def create_batch_request(
    file_id,
    completion_window="24h",
    metadata=None,
):
    """
    Creates a batch processing request for embeddings.

    Args:
        file_id (str): File ID of the uploaded .jsonl file.
        completion_window (str): Time window for the job to complete (e.g., "24h").
        metadata (dict, optional): Metadata dictionary for additional job details.

    Returns:
        str: Batch ID assigned by OpenAI.

    Raises:
        Exception: If there is an error during batch creation.
    """
    if metadata is None:
        metadata = {
            "description": "Default batch job for embedding text content marine engineering manual pages"
        }

    try:
        response = openai.batches.create(
            input_file_id=file_id,
            endpoint="/v1/embeddings",
            completion_window=completion_window,
            metadata=metadata,
        )
        batch_id = response.id
        logging.info(f"Embedding batch request created. Batch ID: {batch_id}")
        print(f"Embedding batch request created. Batch ID: {batch_id}")
        return batch_id
    except Exception as e:
        logging.error(f"Error creating embedding batch request: {e}")
        raise


def monitor_batch(batch_id):
    """
    Monitors the status of a batch processing request.

    Args:
        batch_id (str): Batch ID assigned by OpenAI.

    Returns:
        str: Result File ID if processing is completed.

    Raises:
        Exception: If the batch processing fails or the job status is not completed.
    """
    try:
        logging.info(f"Monitoring batch with ID: {batch_id}")
        print("Monitoring batch status...")
        while True:
            status = openai.batches.retrieve(batch_id)
            # dd(status)
            current_status = status.status  # Use dot notation to access status
            print(f"Current status: {current_status}")

            if current_status == "validating":
                print("The input file is being validated...")
            elif current_status == "failed":
                logging.error("Batch validation failed!")
                raise Exception("Batch validation failed!")
            elif current_status == "in_progress":
                print("Batch is currently being processed...")
            elif current_status == "finalizing":
                print("Batch has completed; results are being prepared...")
            elif current_status == "completed":
                result_file_id = status.output_file_id
                if result_file_id:
                    logging.info(
                        f"Embedding batch processing completed. Result File ID: {result_file_id}"
                    )
                    print(
                        f"Embedding batch processing completed! Result File ID: {result_file_id}"
                    )
                    return result_file_id
                else:
                    logging.warning("Batch completed but no output file was generated.")
                    raise Exception("No output file generated.")
            elif current_status == "expired":
                logging.error(
                    "Batch expired: Could not complete within the 24-hour time window."
                )
                raise Exception(
                    "Batch expired: Could not complete within the 24-hour time window."
                )
            elif current_status == "cancelling":
                print("Batch is being cancelled...")
            elif current_status == "cancelled":
                logging.warning("Batch was cancelled.")
                raise Exception("Batch was cancelled.")

            # Wait before checking the status again
            time.sleep(30)
    except Exception as e:
        logging.error(f"Error monitoring batch: {e}")
        raise


def download_batch_results(result_file_id, output_file):
    """
    Downloads the results of a completed embedding batch request.

    Args:
        result_file_id (str): Result File ID assigned by OpenAI.
        output_file (str): Path to save the results file.

    Raises:
        Exception: If there is an error during download.
    """
    try:
        logging.info(
            f"Downloading raw batch results for Result File ID: {result_file_id}"
        )
        print(f"Downloading raw batch results from Result File ID: {result_file_id}...")

        # Get file content using the result file ID
        result_file = openai.files.content(result_file_id)

        # Write the binary content to the output file
        with open(
            output_file,
            "wb",
        ) as f:
            f.write(result_file.read())

        logging.info(f"Raw batch results saved successfully: {output_file}")
        print(f"Raw batch results saved successfully to: {output_file}")
    except Exception as e:
        logging.error(f"Error downloading batch results: {e}")
        raise


def save_responses_as_json(
    result_file,
    input_folder,
    output_folder="./data/vectors",
):
    """
    Extracts embeddings from the batch output file and saves them as a JSON file.
    The JSON file will be named after the basename of the input folder.

    Args:
        result_file (str): Path to the batch output file (embeddings_batch_output.jsonl or other user-defined .jsonl).
        input_folder (str): The folder where the input Markdown files are stored.
        output_folder (str): Directory where the extracted embeddings will be saved.

    Raises:
        Exception: If an error occurs during file processing.
    """
    try:
        logging.info(f"Extracting embeddings from: {result_file}")
        print(f"Extracting embeddings from batch output: {result_file}")

        # Ensure input_folder is correctly processed to get its basename
        base_name = os.path.basename(os.path.abspath(input_folder))
        output_filename = f"{base_name}.json"  # Use input folder name for output file

        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)

        extracted_embeddings = []

        with open(result_file, "r", encoding="utf-8") as infile:
            for line in infile:
                try:
                    response = json.loads(line)

                    if response.get("response", {}).get("status_code") == 200:
                        data_list = response["response"]["body"].get("data", [])

                        for item in data_list:
                            if item.get("object") == "embedding":
                                extracted_embeddings.append(
                                    {
                                        "custom_id": response.get(
                                            "custom_id", "unknown_id"
                                        ),
                                        "embedding": item.get("embedding", []),
                                        "model": response["response"]["body"].get(
                                            "model", "unknown"
                                        ),
                                        "index": item.get("index", 0),
                                        "usage": response["response"]["body"].get(
                                            "usage", {}
                                        ),
                                    }
                                )
                    else:
                        logging.warning(
                            f"Skipping entry due to non-200 status: {response.get('response', {}).get('status_code')}"
                        )

                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing JSON from batch output: {e}")

        # Define the output file path with dynamic naming
        output_file_path = os.path.join(output_folder, output_filename)

        # Save extracted embeddings to a JSON file
        with open(output_file_path, "w", encoding="utf-8") as outfile:
            json.dump(extracted_embeddings, outfile, indent=4)

        logging.info(f"Processed embeddings saved in JSON format: {output_file_path}")
        print(f"\nProcessed embeddings saved in JSON format: {output_file_path}")

    except Exception as e:
        logging.error(f"Error saving responses as JSON: {e}")
        print(f"An error occurred: {e}")


def cancel_batch(batch_id):
    """
    Cancels an ongoing batch.

    Args:
        batch_id (str): Batch ID assigned by OpenAI.

    Raises:
        Exception: If there is an error during cancellation.
    """
    confirm = (
        input(f"Are you sure you want to cancel Batch ID {batch_id}? (yes/no): \t")
        .strip()
        .lower()
    )
    if confirm == "yes":
        try:
            openai.batches.cancel(batch_id)
            logging.info(f"Batch with ID {batch_id} is being cancelled.")
            print(
                f"Batch with ID {batch_id} is being cancelled. It may take up to 10 minutes to complete the cancellation process."
            )
        except Exception as e:
            logging.error(f"Error cancelling batch with ID {batch_id}: {e}")
            raise
    else:
        print("Cancellation aborted.")


def list_batches(url="/v1/embeddings", limit=10, after=None):
    """
    Lists only the batches belonging to the same API as the user-defined url parameter with optional pagination and allows copying the full batch ID.

    Args:
        url (str): The current OpenAI API endpoint (default: "/v1/embeddings").
        limit (int): Number of batches to list per page (default: 10).
        after (str, optional): Cursor for pagination to fetch results after a specific batch.

    Returns:
        None
    """
    try:
        params = {"limit": limit}
        if after:
            params["after"] = after

        # Retrieve batches from OpenAI API with pagination
        response = openai.batches.list(**params)

        # Extract only relevant batches
        batches = [batch for batch in response if batch.endpoint == url]

        if not batches:
            print(f"\nNo batches found for {url}.\n")
            return

        # Store batch IDs for clipboard functionality
        batch_ids = {}

        # Display header
        print(f"\n=== Embedding Batches for \033[96m{url}\033[0m ===")
        print(
            f"{'Index':<6} {'Batch ID (truncated)':<20} {'Status':<15} {'Created At':<20} {'Metadata':<30}"
        )
        print("-" * 100)

        # Display filtered batch data
        for i, batch in enumerate(batches, start=1):
            truncated_id = batch.id[:8] + "..." + batch.id[-8:]  # Truncate the ID
            metadata_desc = (
                batch.metadata.get("description", "N/A") if batch.metadata else "N/A"
            )
            created_at_formatted = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.gmtime(batch.created_at)
            )
            print(
                f"{i:<6} {truncated_id:<20} {batch.status:<15} {created_at_formatted:<20} {metadata_desc:<30}"
            )
            batch_ids[i] = batch.id  # Store the full batch ID for each index

            # Stop after reaching the limit BECAUSE FOR SOME REASON API DOES NOT WORK??!
            if i >= limit:
                break

        print("-" * 100)

        # Display the next cursor for pagination
        next_cursor = getattr(response, "after", None)
        if next_cursor:
            print(f"Next Cursor: {next_cursor}\n")
        else:
            print("No more batches.\n")

        # Allow user to copy a batch ID
        choice = input(
            "Enter the index of the batch to copy its full ID (or press Enter to skip): \t"
        ).strip()
        if choice.isdigit():
            index = int(choice)
            if index in batch_ids:
                full_id = batch_ids[index]
                pyperclip.copy(full_id)
                print(f"Full Batch ID copied to clipboard: {full_id}")
            else:
                print("Invalid index. No ID copied.")
        else:
            print("No ID selected for copying.")

    except Exception as e:
        logging.error(f"Error listing embedding batches: {e}")
        print(f"An error occurred: {e}")


def get_openai_balance():
    """
    Retrieves and displays the current OpenAI API balance, or explains the restriction if unavailable.

    Returns:
        dict: A dictionary containing total and remaining credits, or None if the balance cannot be retrieved.
    """
    try:
        # OpenAI API endpoint for billing/credits
        url = "https://api.openai.com/v1/dashboard/billing/credit_grants"
        headers = {
            "Authorization": f"Bearer {openai.api_key}",
            "Content-Type": "application/json",
        }

        # Make the GET request
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Parse the response JSON
        data = response.json()
        total_granted = data.get("total_granted", 0.0)
        total_used = data.get("total_used", 0.0)
        total_available = total_granted - total_used

        # Display the balance information
        print("\n=== OpenAI API Balance ===")
        print(f"Total Granted Credits: ${total_granted:.2f}")
        print(f"Total Used Credits: ${total_used:.2f}")
        print(f"Total Available Credits: ${total_available:.2f}")
        print("==========================\n")
        return {
            "total_granted": total_granted,
            "total_used": total_used,
            "total_available": total_available,
        }

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401 or "session key" in e.response.text:
            print("\n=== Balance Retrieval Restricted ===")
            print(
                "The OpenAI API key you are using cannot access balance information because the "
                "endpoint /v1/dashboard/billing/credit_grants is restricted to session keys. "
                "Session keys are used for browser-based contexts where user authentication occurs.\n"
            )
            print(
                "At the moment, OpenAI does not provide an endpoint for balance retrieval using a "
                "server-side secret API key. This functionality may be updated or introduced in the future.\n"
            )
            print(
                "To check your balance, please log in to the OpenAI Dashboard and navigate to the "
                "Billing or Usage section:\n"
                "https://platform.openai.com/account/usage\n"
            )
            print("==========================\n")
        else:
            logging.error(f"Error retrieving API balance: {e}")
            print(f"An error occurred while retrieving the balance: {e}")
        return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Error retrieving API balance: {e}")
        print(f"An error occurred while retrieving the balance: {e}")
        return None


def count_tokens(text: str, model: str) -> int:
    """
    Counts the number of tokens in a given text string based on the model's encoding.

    Args:
        text (str): The input text to tokenize.
        model (str): The OpenAI model being used.

    Returns:
        int: The number of tokens in the input text.
    """
    try:
        # Dynamically determine the encoding based on the model name
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Default to cl100k_base if the model is unknown (as it covers many models)
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def view_log_file(log_file="embeddings_openai.log"):
    """
    Allows the user to view the last N lines of the log file.

    Args:
        log_file (str): Path to the log file (default: "ocr_openai.log").
    """
    if not os.path.exists(log_file):
        print(f"Log file '{log_file}' does not exist.")
        return

    try:
        # Ask user for the number of lines to display
        num_lines_input = input(
            "\nEnter the number of log lines to display (press Enter for last 10): \t"
        ).strip()
        num_lines = int(num_lines_input) if num_lines_input.isdigit() else 10

        with open(log_file, "r", encoding="utf-8") as file:
            # Read last num_lines using deque
            last_lines = collections.deque(file, num_lines)

        print("\n=== Log Output ===\n")
        for line in last_lines:
            print(line.strip())

    except Exception as e:
        print(f"An error occurred while reading the log file: {e}")


def list_uploaded_files():
    """
    Lists batch-related files uploaded to OpenAI with user-specified sorting order and limit.
    Allows copying a file ID to the clipboard.
    """
    try:
        # Prompt the user for limit, ensuring it's within 1-10,000
        while True:
            limit_input = input(
                "Enter the number of files to retrieve (1-10,000, default 100): "
            ).strip()
            if not limit_input:
                limit = 100  # Default value
                break
            if limit_input.isdigit():
                limit = int(limit_input)
                if 1 <= limit <= 10000:
                    break
            print("Invalid input. Please enter a number between 1 and 10,000.")

        # Prompt the user for sorting order
        while True:
            order = (
                input(
                    "Enter sorting order (asc for oldest first, desc for newest first, default: desc): "
                )
                .strip()
                .lower()
            )
            if order in ["asc", "desc", ""]:
                order = order if order else "desc"  # Default to desc
                break
            print("Invalid input. Please enter 'asc' or 'desc'.")

        params = {
            "purpose": "batch",  # Always filter by batch files
            "limit": limit,
            "order": order,
        }

        response = openai.files.list(**params)
        files = response.data

        if not files:
            print("\nNo batch files found.")
            return

        # Store file IDs for clipboard functionality
        file_ids = {}

        print(f"\n=== Batch Files Uploaded to OpenAI (Order: {order.upper()}) ===")
        print(
            f"{'Index':<6} {'File ID (truncated)':<25} {'Filename':<25} {'Size (Bytes)':<15} {'Created At'}"
        )
        print("-" * 100)

        for i, file in enumerate(files[:limit], start=1):
            truncated_id = file.id[:8] + "..." + file.id[-8:]  # Truncate for display
            file_size_mb = file.bytes / (1024 * 1024)  # Convert bytes to MB
            created_at = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.gmtime(file.created_at)
            )
            print(
                f"{i:<6} {truncated_id:<25} {file.filename:<25} {file_size_mb:<15.2f} {created_at}"
            )
            file_ids[i] = file.id  # Store the full file ID for each index

        print("-" * 100)

        # Allow user to copy a file ID
        choice = input(
            "Enter the index of the file to copy its full ID (or press Enter to skip): "
        ).strip()
        if choice.isdigit():
            index = int(choice)
            if index in file_ids:
                full_id = file_ids[index]
                pyperclip.copy(full_id)
                print(f"Full File ID copied to clipboard: {full_id}")
            else:
                print("Invalid index. No ID copied.")
        else:
            print("No ID selected for copying.")

    except Exception as e:
        logging.error(f"Error listing batch files: {e}")
        print(f"An error occurred: {e}")


def get_chromadb_client():
    """
    Returns the existing ChromaDB client instance.
    If it is not initialized, it returns None.
    """
    global chromadb_client
    return chromadb_client


def initialize_chromadb(persistent_path=None):
    """
    Initializes the ChromaDB client.
    If an instance is already running, it warns the user before reinitializing.

    Args:
        persistent_path (str or None): Path for persistent storage. If None, initializes in-memory mode.

    Returns:
        chromadb.Client or chromadb.PersistentClient: The ChromaDB client instance.
    """
    global chromadb_client

    # Check if ChromaDB is already active
    if chromadb_client and is_chromadb_active(chromadb_client):
        print("\033[33mWarning:\033[0m A ChromaDB instance is already running.")
        print(
            "Reinitializing may overwrite existing collections or cause unintended behavior."
        )
        confirm = input("Do you want to proceed? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("\nChromaDB initialization aborted.")
            return chromadb_client  # Return existing instance

    if persistent_path and os.path.exists(persistent_path):
        print(f"\nReconnecting to existing ChromaDB instance at: {persistent_path}")
        chromadb_client = chromadb.PersistentClient(path=persistent_path)
    else:
        if persistent_path:
            print("\nInitializing ChromaDB in persistent mode...")
            chromadb_client = chromadb.PersistentClient(path=persistent_path)
        else:
            print("\nInitializing ChromaDB in in-memory mode...")
            chromadb_client = chromadb.Client()

    return chromadb_client


def is_chromadb_active(client):
    """
    Checks if the ChromaDB client is active by calling heartbeat().

    Args:
        client (chromadb.Client or chromadb.PersistentClient): The ChromaDB client instance.

    Returns:
        bool: True if ChromaDB is active, False otherwise.
    """
    try:
        client.heartbeat()  # If this works, the instance is active. Otherwise will raise an exception if inactive
        return True
    except Exception as e:
        logging.warning(f"ChromaDB heartbeat check failed: {e}")
        return False


def connect_to_existing_chromadb():
    """
    Connect to an existing ChromaDB persistent instance and list collections.
    Adds pagination, filtering, sorting, text wrapping, and highlights the last used collection.
    """
    import textwrap
    from math import ceil

    global chromadb_client
    config = load_config()
    default_collection = config.get("chromadb_collection_name", "")

    while True:
        existing_path = input(
            "Enter the path to the existing ChromaDB instance: "
        ).strip()

        if not os.path.exists(existing_path) or not os.path.isdir(existing_path):
            print(
                "\033[31mError:\033[0m The specified path does not exist or is not a directory."
            )
            continue

        if not os.path.exists(os.path.join(existing_path, "chroma.sqlite3")):
            print(
                "\033[31mError:\033[0m Not a valid ChromaDB instance (missing chroma.sqlite3)."
            )
            confirm = input("Do you still want to proceed? (yes/no): ").strip().lower()
            if confirm != "yes":
                return config

        try:
            chromadb_client = chromadb.PersistentClient(path=existing_path)
            print("\033[32mSuccessfully connected to the ChromaDB instance.\033[0m")
        except Exception as e:
            print(f"\033[31mConnection failed:\033[0m {e}")
            return config

        # Gather collection metadata
        collection_data = []
        for name in chromadb_client.list_collections():
            try:
                col = chromadb_client.get_collection(name)
                metadata = col.metadata or {}
            except Exception:
                metadata = {}

            raw_created = metadata.get("created_at", "N/A")
            if raw_created != "N/A":
                try:
                    parsed_dt = datetime.fromisoformat(raw_created)
                    formatted = parsed_dt.strftime("%Y-%m-%d %H:%M:%S (UTC)")
                except Exception:
                    formatted = raw_created
            else:
                formatted = "N/A"

            collection_data.append(
                {
                    "name": name,
                    "distance_function": metadata.get("distance_function", "Unknown"),
                    "description": metadata.get("description", "N/A"),
                    "created_at": formatted,
                    "created_at_raw": raw_created,
                }
            )

        if not collection_data:
            print("\n\033[33mWarning:\033[0m No collections found.")
            confirm = input("Create the default collection? (yes/no): ").strip().lower()
            if confirm == "yes":
                chromadb_client.get_or_create_collection(name=default_collection)
                print(f"Created collection: {default_collection}")
            else:
                return config
            selected_collection = default_collection
            break

        current_data = collection_data[:]
        page = 1
        per_page = 5
        sort_key = None
        reverse_sort = False

        while True:
            clear_screen()

            # Column widths
            name_width = 30
            dist_width = 18
            desc_width = 40
            created_width = max(len(c["created_at"]) for c in current_data) + 2
            buffer = 10

            # Header
            header = f"{'#':<4} {'Name':<{name_width}} {'Distance Function':<{dist_width}} {'Description':<{desc_width}} {'Created At'}"
            total_pages = ceil(len(current_data) / per_page)

            print("\n\033[1mAvailable ChromaDB Collections:\033[0m")
            print("=" * (len(header) + buffer))
            print(header)
            print("=" * (len(header) + buffer))

            start = (page - 1) * per_page
            end = start + per_page

            for i, col in enumerate(current_data[start:end], start=start + 1):
                name = col["name"]
                dist = col["distance_function"]
                desc = col["description"]
                created = col["created_at"]

                name_lines = textwrap.wrap(name, name_width)
                desc_lines = textwrap.wrap(desc, desc_width)
                max_lines = max(len(name_lines), len(desc_lines))
                is_default = name == default_collection

                for line_idx in range(max_lines):
                    n = name_lines[line_idx] if line_idx < len(name_lines) else ""
                    d = desc_lines[line_idx] if line_idx < len(desc_lines) else ""
                    dist_str = dist if line_idx == 0 else ""
                    created_str = created if line_idx == 0 else ""
                    index_str = f"{i:<4}" if line_idx == 0 else "    "

                    row = f"{index_str} {n:<{name_width}} {dist_str:<{dist_width}} {d:<{desc_width}} {created_str:<{created_width}}"
                    if is_default:
                        print(f"\033[36m{row}\033[0m")  # Highlight default
                    else:
                        print(row)

            print("=" * (len(header) + buffer))
            print(f"(Page {page}/{total_pages})")

            # Prompt
            print("\nOptions:")
            print("  [n] Next Page")
            print("  [p] Previous Page")
            print("  [f] Filter")
            print("  [r] Reset Filters")
            print("  [s] Sort")
            print("  [c] Choose Collection")
            print("  [q] Quit to Menu")

            cmd = input("\nEnter your choice: ").strip().lower()

            if cmd == "n":
                if page < total_pages:
                    page += 1
                else:
                    print("You are already on the last page.")
                    input("Press Enter to continue...")
            elif cmd == "p":
                if page > 1:
                    page -= 1
                else:
                    print("You are already on the first page.")
                    input("Press Enter to continue...")
            elif cmd == "f":
                name_filter = input("Filter by name: ").strip().lower()
                date_filter = input("Filter by created after (YYYY-MM-DD): ").strip()
                filtered = collection_data

                if name_filter:
                    filtered = [c for c in filtered if name_filter in c["name"].lower()]
                if date_filter:
                    try:
                        dt = datetime.strptime(date_filter, "%Y-%m-%d")
                        filtered = [
                            c
                            for c in filtered
                            if c["created_at_raw"] != "N/A"
                            and datetime.fromisoformat(c["created_at_raw"]) >= dt
                        ]
                    except ValueError:
                        print("Invalid date format. Skipping date filter.")
                        input("Press Enter to continue...")

                current_data = filtered
                page = 1
            elif cmd == "r":
                current_data = collection_data[:]
                page = 1
            elif cmd == "s":
                print("\nSort by:")
                print("  1. Name")
                print("  2. Distance Function")
                print("  3. Description")
                print("  4. Created At")
                sort_map = {
                    "1": "name",
                    "2": "distance_function",
                    "3": "description",
                    "4": "created_at_raw",
                }
                choice = input("Enter number: ").strip()
                sort_key = sort_map.get(choice)
                if sort_key:
                    reverse_sort = (
                        input("Descending? (yes/no): ").strip().lower() == "yes"
                    )
                    current_data = sorted(
                        current_data,
                        key=lambda x: (
                            datetime.fromisoformat(x[sort_key])
                            if sort_key == "created_at_raw" and x[sort_key] != "N/A"
                            else (
                                x[sort_key].lower()
                                if isinstance(x[sort_key], str)
                                else x[sort_key]
                            )
                        ),
                        reverse=reverse_sort,
                    )
                    page = 1
            elif cmd == "c":
                selection = input(
                    "\nEnter the number of the collection to use: "
                ).strip()
                if selection.isdigit():
                    idx = int(selection)
                    if 1 <= idx <= len(current_data):
                        selected_collection = current_data[idx - 1]["name"]
                        break
                print("Invalid selection. Try again.")
            elif cmd == "q":
                selected_collection = default_collection
                break
            else:
                print("Invalid input.")
                input("Press Enter to continue...")

        # Save config and return
        config["chromadb_storage_mode"] = "persistent"
        config["chromadb_persistence_path"] = existing_path
        config["chromadb_collection_name"] = selected_collection
        save_config(config)

        print(
            f"\n\033[32mChromaDB is now active.\033[0m Using collection: {selected_collection}"
        )
        return config


def get_or_create_collection(client, collection_name):
    """
    Gets an existing collection or creates a new one if it doesn't exist.

    Args:
        client (chromadb.PersistentClient): ChromaDB client instance.
        collection_name (str): The name of the collection to create.

    Returns:
        chromadb.Collection: The ChromaDB collection object.
    """
    return client.get_or_create_collection(collection_name)


def list_chromadb_collections():
    """
    Lists available ChromaDB collections with support for pagination, filtering, and sorting.

    The user can:
    - Navigate between pages
    - Filter collections by name or creation date
    - Sort collections by name, distance function, description, or creation date
    - View full descriptions if they are truncated in the table

    Each collection displays its name, distance function, short description, and creation date.
    """
    client = get_chromadb_client()

    if not client or not is_chromadb_active(client):
        print(
            "\033[31mError:\033[0m ChromaDB is not active. Use Option 19 or 20 first."
        )
        return

    try:
        collection_names = client.list_collections()
        if not collection_names:
            print("\033[33mNo collections found in ChromaDB.\033[0m")
            return

        # Fetch metadata for all collections
        collection_data = []
        for name in collection_names:
            try:
                col = client.get_collection(name)
                metadata = col.metadata if hasattr(col, "metadata") else {}
            except Exception as e:
                logging.warning(f"Failed to get metadata for {name}: {e}")
                metadata = {}

            # Format created_at in human readable format
            raw_created_at = metadata.get("created_at", "N/A")
            if raw_created_at != "N/A":
                try:
                    parsed_dt = datetime.fromisoformat(raw_created_at)
                    formatted_created_at = parsed_dt.strftime("%Y-%m-%d %H:%M:%S (UTC)")

                    # Calculate relative time
                    now_utc = datetime.now(timezone.utc)
                    delta = now_utc - parsed_dt
                    seconds = int(delta.total_seconds())

                    if seconds < 60:
                        ago_str = f"{seconds} seconds ago"
                    elif seconds < 3600:
                        minutes = seconds // 60
                        ago_str = f"{minutes} minute{'s' if minutes != 1 else ''} ago"
                    elif seconds < 86400:
                        hours = seconds // 3600
                        ago_str = f"{hours} hour{'s' if hours != 1 else ''} ago"
                    else:
                        days = seconds // 86400
                        ago_str = f"{days} day{'s' if days != 1 else ''} ago"

                    created_at_display = f"{formatted_created_at} — {ago_str}"
                except Exception:
                    created_at_display = raw_created_at
            else:
                created_at_display = "N/A"
                raw_created_at = None

            collection_data.append(
                {
                    "name": name,
                    "distance_function": metadata.get("distance_function", "Unknown"),
                    "description": metadata.get("description", "N/A"),
                    "created_at": created_at_display,
                    "created_at_raw": raw_created_at,
                }
            )

        def print_table(data, title, page, per_page):
            total_pages = ceil(len(data) / per_page)

            # Define maximum widths for each column
            name_width = 30
            distance_width = 18
            desc_width = 40
            created_width = 40
            buffer = 5  # Space between columns

            # Header line length with buffer spaces
            header_length = (
                5
                + buffer
                + name_width
                + buffer
                + distance_width
                + buffer
                + desc_width
                + buffer
                + created_width
            )

            header = (
                f"{'#':<5}{' ' * buffer}"
                f"{'Name':<{name_width}}{' ' * buffer}"
                f"{'Distance Function':<{distance_width}}{' ' * buffer}"
                f"{'Description':<{desc_width}}{' ' * buffer}"
                f"{'Created At'}"
            )

            print(f"\n\033[1m{title} (Page {page}/{total_pages})\033[0m")
            print("=" * header_length)
            print(header)
            print("=" * header_length)

            start = (page - 1) * per_page
            end = start + per_page
            for i, col in enumerate(data[start:end], start=start + 1):
                wrapped_name = wrap(col["name"], name_width)
                wrapped_desc = wrap(col["description"], desc_width)
                wrapped_created = wrap(col["created_at"], created_width)

                max_lines = max(
                    len(wrapped_name), len(wrapped_desc), len(wrapped_created)
                )

                for line in range(max_lines):
                    print(
                        f"{str(i) if line == 0 else '':<5}{' ' * buffer}"
                        f"{wrapped_name[line] if line < len(wrapped_name) else '':<{name_width}}{' ' * buffer}"
                        f"{col['distance_function'] if line == 0 else '':<{distance_width}}{' ' * buffer}"
                        f"{wrapped_desc[line] if line < len(wrapped_desc) else '':<{desc_width}}{' ' * buffer}"
                        f"{wrapped_created[line] if line < len(wrapped_created) else ''}"
                    )
                print()

            print("=" * header_length)

        ##  Not necessary now with text wrapping
        # def prompt_view_full_descriptions(data):
        #     view_desc = (
        #         input("\nWould you like to view full descriptions? (yes/no): ")
        #         .strip()
        #         .lower()
        #     )
        #     if view_desc == "yes":
        #         print("\n\033[1mFull Descriptions:\033[0m")
        #         print("=" * 80)
        #         for i, col in enumerate(data, 1):
        #             print(f"{i}. \033[96m{col['name']}\033[0m - {col['description']}")
        #         print("=" * 80)

        current_data = collection_data[:]
        page = 1
        per_page = 5

        while True:
            clear_screen()
            print_table(current_data, "Available ChromaDB Collections", page, per_page)
            # prompt_view_full_descriptions(current_data)

            print("\nOptions:")
            print("  [n] Next Page")
            print("  [p] Previous Page")
            print("  [f] Filter")
            print("  [r] Reset Filters")
            print("  [s] Sort")
            print("  [q] Quit")

            cmd = input("\nEnter your choice: ").strip().lower()

            if cmd == "n":
                if page * per_page < len(current_data):
                    page += 1
                else:
                    print("You are already on the last page.")
                    input("Press Enter to continue...")
            elif cmd == "p":
                if page > 1:
                    page -= 1
                else:
                    print("You are already on the first page.")
                    input("Press Enter to continue...")
            elif cmd == "f":
                name_filter = input("Filter by name (substring): ").strip().lower()
                date_filter = input(
                    "Filter by created after date (YYYY-MM-DD): "
                ).strip()
                filtered = collection_data

                if name_filter:
                    filtered = [c for c in filtered if name_filter in c["name"].lower()]
                if date_filter:
                    try:
                        dt = datetime.strptime(date_filter, "%Y-%m-%d")
                        filtered = [
                            c
                            for c in filtered
                            if c["created_at"] != "N/A"
                            and c["created_at_raw"] != "N/A"
                            and datetime.fromisoformat(c["created_at_raw"]) >= dt
                        ]
                    except ValueError:
                        print(
                            "\033[33mInvalid date format. Skipping date filter.\033[0m"
                        )

                current_data = filtered
                page = 1
            elif cmd == "r":
                current_data = collection_data[:]
                page = 1
            elif cmd == "s":
                print("\nSort by:")
                print("  1. Name")
                print("  2. Distance Function")
                print("  3. Description")
                print("  4. Created At")
                sort_field_map = {
                    "1": "name",
                    "2": "distance_function",
                    "3": "description",
                    "4": "created_at_raw",
                }
                sort_choice = input("Enter number: ").strip()
                sort_key = sort_field_map.get(sort_choice)
                if sort_key:
                    order = (
                        input("Ascending or descending? (asc/desc): ").strip().lower()
                    )
                    reverse = order == "desc"
                    current_data = sorted(
                        current_data,
                        key=lambda x: (
                            datetime.fromisoformat(x["created_at_raw"])
                            if sort_key == "created_at_raw"
                            and x["created_at_raw"] != "N/A"
                            else (
                                x.get(sort_key, "").lower()
                                if isinstance(x.get(sort_key), str)
                                else x.get(sort_key)
                            )
                        ),
                        reverse=reverse,
                    )
                    page = 1
                else:
                    print("Invalid sort option.")
                    input("Press Enter to continue...")
            elif cmd == "q":
                clear_screen()
                break
            else:
                print("Invalid input.")
                input("Press Enter to continue...")

    except Exception as e:
        print(f"\033[31mError listing ChromaDB collections:\033[0m {e}")
        logging.error(f"Error listing collections: {e}")


def create_new_chromadb_collection():
    """
    Creates a new ChromaDB collection and optionally switches to it.
    Returns updated config if a switch occurs.
    """
    client = get_chromadb_client()
    config = load_config()

    if not client or not is_chromadb_active(client):
        print(
            "\033[31mError:\033[0m ChromaDB is not active. Use Option 19 or 20 first."
        )
        logging.warning("Attempted to create a collection while ChromaDB is inactive.")
        return config

    existing_collections = client.list_collections()
    existing_names = (
        existing_collections
        if isinstance(existing_collections[0], str)
        else [col.name for col in existing_collections]
    )

    print("\n\033[1m=== Create a New ChromaDB Collection ===\033[0m")
    collection_name = input("> Enter a name for the new collection: ").strip()
    if not collection_name:
        print("\033[31mCollection name cannot be empty.\033[0m")
        logging.warning("User attempted to create a collection with an empty name.")
        return config

    try:
        existing_names = client.list_collections()
        if collection_name in existing_names:
            print(f"\n\033[33mCollection '{collection_name}' already exists.\033[0m")
            switch = (
                input("> Do you want to switch to it now? (yes/no): ").strip().lower()
            )
            if switch == "yes":
                config["chromadb_collection_name"] = collection_name
                save_config(config)
                print(f"\n\033[36mNow using collection:\033[0m {collection_name}")
            return config

        # Prompt for distance function
        print("\n> Select Distance Function:")
        print("  1. Cosine (Recommended for OpenAI Embeddings)")
        print("  2. Euclidean")
        print("  3. Dot Product")
        dist_choice = input("> Enter 1, 2, or 3: ").strip()
        dist_map = {"1": "cosine", "2": "euclidean", "3": "dot_product"}
        distance_function = dist_map.get(dist_choice, "cosine")

        # Prompt for collection description
        description = input(
            "\n> Enter a short description for this collection: "
        ).strip()
        if not description:
            description = "No description provided"

        # Capture creation time
        created_at = datetime.now(timezone.utc).isoformat()

        # Create collection with enriched metadata
        client.create_collection(
            name=collection_name,
            metadata={
                "distance_function": distance_function,
                "description": description,
                "created_at": created_at,
            },
        )
        print(f"\n\033[32mCollection '{collection_name}' created successfully.\033[0m")
        print(f"\033[90m→ Distance Function:\033[0m {distance_function}")
        print(f"\033[90m→ Description:\033[0m {description}")
        print(f"\033[90m→ Created At:\033[0m {created_at}")

        # Ask user if they want to use this collection
        switch = (
            input("> Do you want to use this collection now? (yes/no): ")
            .strip()
            .lower()
        )
        if switch == "yes":
            config["chromadb_collection_name"] = collection_name
            save_config(config)
            print(f"\n\033[36mNow using collection:\033[0m {collection_name}")

    except Exception as e:
        print(f"\n\033[31mFailed to create or switch collection:\033[0m {e}")
        logging.error(f"Collection creation error: {e}")

    return config


def switch_chromadb_collection():
    """
    Allows the user to switch to another existing ChromaDB collection.
    Returns updated config.
    """
    client = get_chromadb_client()
    config = load_config()

    if not client or not is_chromadb_active(client):
        print(
            "\033[31mError:\033[0m ChromaDB is not active. Use Option 19 or 20 first."
        )
        logging.warning("Attempted to switch collections while ChromaDB is inactive.")
        return config

    try:
        collection_names = client.list_collections()

        if not collection_names:
            print("\033[33mNo collections found in ChromaDB.\033[0m")
            logging.info("No collections available to switch to.")
            return config

        print("\n\033[1mAvailable ChromaDB Collections:\033[0m")
        print("=" * 40)
        for i, name in enumerate(collection_names, 1):
            print(f"{i}. {name}")
        print("=" * 40)

        choice = input("Enter the number of the collection to switch to: ").strip()

        if not choice.isdigit() or not (1 <= int(choice) <= len(collection_names)):
            print("\033[33mInvalid selection.\033[0m")
            return config

        selected_collection = collection_names[int(choice) - 1]
        config["chromadb_collection_name"] = selected_collection
        save_config(config)
        print(f"\n\033[36mNow using collection:\033[0m {selected_collection}")
        logging.info(f"Switched to ChromaDB collection: {selected_collection}")

    except Exception as e:
        print(f"\033[31mAn error occurred:\033[0m {e}")
        logging.error(f"Error during collection switch: {e}")

    return config


def delete_chromadb_collection():
    """
    Allows the user to delete an existing ChromaDB collection.
    Ensures confirmation and prevents deleting the currently active collection.
    Returns updated config if changes occur.
    """
    client = get_chromadb_client()
    config = load_config()

    if not client or not is_chromadb_active(client):
        print(
            "\033[31mError:\033[0m ChromaDB is not active. Use Option 19 or 20 first."
        )
        logging.warning("Attempted to delete a collection while ChromaDB is inactive.")
        return config

    try:
        collection_names = client.list_collections()

        if not collection_names:
            print("\033[33mNo collections found to delete.\033[0m")
            return config

        print("\n\033[1mAvailable Collections:\033[0m")
        print("=" * 40)
        for i, name in enumerate(collection_names, 1):
            print(f"{i}. {name}")
        print("=" * 40)

        choice = input("Enter the number of the collection to delete: ").strip()

        if not choice.isdigit() or not (1 <= int(choice) <= len(collection_names)):
            print("\033[33mInvalid selection.\033[0m")
            return config

        selected_collection = collection_names[int(choice) - 1]

        if selected_collection == config.get("chromadb_collection_name"):
            print(
                f"\033[33mWarning:\033[0m You are attempting to delete the currently active collection: \033[1m{selected_collection}\033[0m."
            )
            print("You must switch to another collection before deleting this one.")
            return config

        confirm = (
            input(
                f"\nAre you sure you want to permanently delete collection '{selected_collection}'? (yes/no): "
            )
            .strip()
            .lower()
        )

        if confirm != "yes":
            print("\n\033[33mDeletion aborted by user.\033[0m")
            return config

        client.delete_collection(name=selected_collection)
        print(
            f"\n\033[32mCollection '{selected_collection}' deleted successfully.\033[0m"
        )
        logging.info(f"Deleted ChromaDB collection: {selected_collection}")

    except Exception as e:
        print(f"\033[31mFailed to delete collection:\033[0m {e}")
        logging.error(f"Error deleting ChromaDB collection: {e}")

    return config


def list_available_embedding_files(output_path="./data/vectors", per_page=5):
    """
    Lists available embedding JSON files from the output folder with metadata,
    including file size and creation date. Supports pagination, filtering, and sorting.
    """
    config = load_config()
    output_folder = config.get("output_folder", "./data/vectors")

    if not os.path.exists(output_folder):
        print(f"\033[31mError:\033[0m Output folder '{output_folder}' does not exist.")
        return

    files = [f for f in os.listdir(output_folder) if f.endswith(".json")]

    if not files:
        print(f"\033[33mNo embedding files found in '{output_folder}'.\033[0m")
        return

    # Load ChromaDB collection metadata to determine loaded files
    client = get_chromadb_client()
    if not client or not is_chromadb_active(client):
        print("\033[31mError:\033[0m ChromaDB is not active.")
        return

    collection_name = config.get("chromadb_collection_name")
    collection = get_or_create_collection(client, collection_name)

    # Get loaded source_file paths from ChromaDB collection metadata
    try:
        results = collection.get(include=["metadatas"])
        loaded_paths = set(
            meta.get("source_file")
            for meta in results.get("metadatas", [])
            if "source_file" in meta
        )
        normalized_loaded_paths = {
            os.path.abspath(os.path.normpath(p)) for p in loaded_paths
        }
    except Exception as e:
        logging.error(f"Failed to fetch metadata from ChromaDB: {e}")
        normalized_loaded_paths = set()

    file_data = []
    for filename in files:
        # full_path = os.path.join(output_folder, filename)
        full_path = os.path.abspath(os.path.join(output_folder, filename))
        try:
            stat = os.stat(full_path)
            size_kb = stat.st_size / 1024
            created = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
            created_str = created.strftime("%Y-%m-%d %H:%M:%S (UTC)")
            delta = datetime.now(timezone.utc) - created

            # Human-readable time delta
            if delta.days >= 1:
                relative = f"[{delta.days} day{'s' if delta.days > 1 else ''} ago]"
            elif delta.seconds >= 3600:
                hours = delta.seconds // 3600
                relative = f"[{hours} hour{'s' if hours > 1 else ''} ago]"
            elif delta.seconds >= 60:
                minutes = delta.seconds // 60
                relative = f"[{minutes} min ago]"
            else:
                relative = "[Just now]"

            created_str = f"{created_str} {relative}"
        except Exception as e:
            logging.warning(f"Failed to get metadata for {filename}: {e}")
            size_kb = 0
            created_str = "Unknown"

        is_loaded = (
            os.path.abspath(os.path.normpath(full_path)) in normalized_loaded_paths
        )

        file_data.append(
            {
                "name": filename,
                "size_kb": size_kb,
                "created": created_str,
                "full_path": full_path,
                "loaded": is_loaded,
            }
        )

    def print_table(data, page, per_page):

        ANSI_GREEN = "\033[92m"
        ANSI_RESET = "\033[0m"

        def strip_ansi(s):
            """
            Useful because ANSI Escape Characters count as string and ruin the row alignments
            """
            return re.sub(r"\x1b\[[0-9;]*m", "", s)

        total_pages = ceil(len(data) / per_page)

        print("\n\033[1mAvailable Embedding Files:\033[0m")
        print(f"[Location: {output_folder}]")
        print(f"\nFound {len(data)} embedding file(s):")

        # Layout configuration
        name_width = 55
        size_width = 14
        created_width = 25
        buffer = 6
        total_width = name_width + size_width + created_width + 5 + buffer

        header = f"{'#':<5} {'Filename':<{name_width}}  {'Size (KB)':<{size_width}}  {'Created At':<{created_width}}"
        print("\n" + "=" * total_width)
        print(header)
        print("=" * total_width)

        start = (page - 1) * per_page
        end = start + per_page
        for i, item in enumerate(data[start:end], start=start + 1):
            # filename =  (" \033[32m[✔]\033[0m" if item["loaded"] else "") + item["name"]
            raw_filename = item["name"]
            wrapped_name = wrap(raw_filename, width=name_width)
            size_str = f"{item['size_kb']:.2f} KB"
            created_str = item["created"]

            # First line
            # Format with color if loaded
            if item["loaded"]:
                print(
                    f"{ANSI_GREEN}{i:<{5}} {wrapped_name[0]:<{name_width}} {size_str:<{size_width}} {created_str:<{created_width}}{ANSI_RESET}"
                )
            else:
                print(
                    f"{i:<{5}} {wrapped_name[0]:<{name_width}} {size_str:<{size_width}} {created_str:<{created_width}}"
                )

            # Remaining wrapped lines
            for line in wrapped_name[1:]:
                if item["loaded"]:
                    print(f"{ANSI_GREEN}{'':<{5}} {line:<{name_width}}{ANSI_RESET}")
                else:
                    print(f"{'':<{5}} {line:<{name_width}}")

            print()  # Extra space between rows

        print("=" * total_width)
        print(f"(Page {page}/{total_pages})")

    current_data = file_data[:]
    page = 1

    while True:
        clear_screen()
        print_table(current_data, page, per_page)

        print("\nOptions:")
        print("  [n] Next Page")
        print("  [p] Previous Page")
        print("  [f] Filter")
        print("  [r] Reset Filters")
        print("  [s] Sort")
        print("  [q] Quit")

        cmd = input("\nEnter your choice: ").strip().lower()

        if cmd == "n":
            if page * per_page < len(current_data):
                page += 1
            else:
                print("You are already on the last page.")
                input("Press Enter to continue...")
        elif cmd == "p":
            if page > 1:
                page -= 1
            else:
                print("You are already on the first page.")
                input("Press Enter to continue...")
        elif cmd == "f":
            name_filter = input("Filter by filename (substring): ").strip().lower()
            date_filter = input("Filter by created after date (YYYY-MM-DD): ").strip()
            print("\nFilter by loaded status:")
            print("  1. Only loaded files")
            print("  2. Only not-loaded files")
            print("  3. Show all (no filter)")
            status_filter = input("Choose [1/2/3]: ").strip()

            filtered = file_data

            if name_filter:
                filtered = [f for f in filtered if name_filter in f["name"].lower()]
            if date_filter:
                try:
                    dt = datetime.strptime(date_filter, "%Y-%m-%d")
                    filtered = [
                        f
                        for f in filtered
                        if f["created"] != "Unknown"
                        and datetime.strptime(f["created"], "%Y-%m-%d %H:%M:%S (UTC)")
                        >= dt
                    ]
                except ValueError:
                    print("\033[33mInvalid date format. Skipping date filter.\033[0m")

            if status_filter == "1":
                filtered = [f for f in filtered if f["loaded"]]
            elif status_filter == "2":
                filtered = [f for f in filtered if not f["loaded"]]
            elif status_filter == "3":
                pass
            else:
                print("\033[33mInvalid status filter. Showing all files.\033[0m")

            current_data = filtered
            page = 1
        elif cmd == "r":
            current_data = file_data[:]
            page = 1
        elif cmd == "s":
            print("\nSort by:")
            print("  1. Filename")
            print("  2. Size")
            print("  3. Created At")
            sort_choice = input("Enter number: ").strip()
            sort_map = {
                "1": "name",
                "2": "size_kb",
                "3": "created",
            }
            key = sort_map.get(sort_choice)
            if key:
                reverse = (
                    input("Ascending or descending? (asc/desc): ").strip().lower()
                    == "desc"
                )
                current_data = sorted(
                    current_data,
                    key=lambda x: x[key].lower() if isinstance(x[key], str) else x[key],
                    reverse=reverse,
                )
                page = 1
            else:
                print("Invalid sort option.")
                input("Press Enter to continue...")
        elif cmd == "q":
            clear_screen()
            break
        else:
            print("Invalid input.")
            input("Press Enter to continue...")


def load_embeddings_into_chromadb(
    file_path, collection, use_batch=True, batch_size=100
):
    """
    Loads pre-generated embeddings from a JSON file into ChromaDB with optional batch insertion.

    Args:
        file_path (str): Path to the JSON file containing embeddings.
        collection (chromadb.Collection): ChromaDB collection to store the embeddings.
        use_batch (bool): Whether to use batch insertion.
        batch_size (int): Size of each batch for batch insertion.

    Raises:
        Exception: If the JSON file cannot be read or processed.
    """
    try:
        start_time = datetime.now()
        logging.info(
            f"Started loading embeddings from '{file_path}' into collection '{collection.name}'"
        )
        print(f"\nLoading file: {os.path.basename(file_path)}")
        print(f"Target collection: {collection.name}")
        print(f"Insertion mode: {'Batch' if use_batch else 'Single'}")
        if use_batch:
            print(f"Batch size: {batch_size}")
        print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        with open(file_path, "r", encoding="utf-8") as file:
            embeddings = json.load(file)

        total_tokens = 0
        total = len(embeddings)
        progress = tqdm(total=total, desc="Inserting Embeddings", ncols=100)

        if use_batch:
            batch_ids, batch_embeddings, batch_metadatas = [], [], []

            for item in embeddings:
                try:
                    batch_ids.append(item["custom_id"])
                    batch_embeddings.append(item["embedding"])
                    metadata = {
                        "model": item.get("model", "unknown"),
                        "token_count": item["usage"].get("total_tokens", 0),
                        "source_file": file_path,
                    }
                    total_tokens += metadata["token_count"]
                    batch_metadatas.append(metadata)
                except Exception as e:
                    logging.warning(f"Skipping malformed embedding item: {e}")
                    continue

                if len(batch_ids) >= batch_size:
                    collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas,
                    )
                    progress.update(len(batch_ids))
                    logging.info(f"Inserted batch of {len(batch_ids)} embeddings.")
                    batch_ids, batch_embeddings, batch_metadatas = [], [], []

            if batch_ids:
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                )
                progress.update(len(batch_ids))
                logging.info(f"Inserted final batch of {len(batch_ids)} embeddings.")

        else:
            for item in embeddings:
                try:
                    collection.add(
                        ids=[item["custom_id"]],
                        embeddings=[item["embedding"]],
                        metadatas=[
                            {
                                "model": item.get("model", "unknown"),
                                "token_count": item["usage"].get("total_tokens", 0),
                                "source_file": file_path,
                            }
                        ],
                    )
                    total_tokens += item["usage"].get("total_tokens", 0)
                    progress.update(1)
                except Exception as e:
                    logging.warning(f"Skipping failed individual insertion: {e}")
                    continue

        progress.close()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(
            f"\nSuccessfully loaded {total} embeddings into collection '{collection.name}'."
        )
        print(f"Total token count: {total_tokens}")
        print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed time: {duration:.2f} seconds")

        logging.info(
            f"Finished loading {total} embeddings into collection '{collection.name}' "
            f"from '{file_path}' | Total tokens: {total_tokens} | Duration: {duration:.2f} sec"
        )

    except Exception as e:
        logging.error(f"Failed to load embeddings from '{file_path}': {e}")
        print(f"\nError loading embeddings from {file_path}: {e}")
        raise e


def load_embeddings_into_chromadb_ui(client, config):
    """
    Interactive CLI for loading embeddings into a ChromaDB collection.
    Includes listing, filtering, sorting, pagination, and file selection.

    Args:
        client (chromadb.PersistentClient): The ChromaDB client instance.
        config (dict): Loaded configuration containing output_folder and collection name.
    """
    output_folder = config.get("output_folder", "./data/vectors")
    collection_name = config.get("chromadb_collection_name")

    if not client or not is_chromadb_active(client):
        print(
            "\033[31mError:\033[0m ChromaDB is not active. Use Option 19 or 20 first."
        )
        return

    # Redundant -Remove
    if not os.path.exists(output_folder):
        print(f"\033[31mError:\033[0m Output folder '{output_folder}' does not exist.")
        return

    embedding_files = [f for f in os.listdir(output_folder) if f.endswith(".json")]
    if not embedding_files:
        print(f"\033[33mNo embedding files found in '{output_folder}'.\033[0m")
        input("Press Enter to return to the menu...")
        return

    # Fetch loaded file paths - For marking already loaded embedding files in selected collection
    try:
        collection = get_or_create_collection(client, collection_name)
        results = collection.get(include=["metadatas"])
        loaded_paths = {
            os.path.abspath(os.path.normpath(meta["source_file"]))
            for meta in results.get("metadatas", [])
            if "source_file" in meta
        }
    except Exception as e:
        logging.error(f"Failed to fetch metadata from ChromaDB: {e}")
        loaded_paths = set()

    # Gather metadata
    file_data = []
    for filename in embedding_files:
        path = os.path.join(output_folder, filename)
        try:
            stat = os.stat(path)
            size_kb = stat.st_size / 1024
            created = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
            created_str = created.strftime("%Y-%m-%d %H:%M:%S (UTC)")
            delta = datetime.now(timezone.utc) - created

            # Human-readable time delta
            if delta.days >= 1:
                relative = f"[{delta.days} day{'s' if delta.days > 1 else ''} ago]"
            elif delta.seconds >= 3600:
                hours = delta.seconds // 3600
                relative = f"[{hours} hour{'s' if hours > 1 else ''} ago]"
            elif delta.seconds >= 60:
                minutes = delta.seconds // 60
                relative = f"[{minutes} min ago]"
            else:
                relative = "[Just now]"

            created_str = f"{created_str} {relative}"
        except Exception as e:
            logging.warning(f"Failed to get metadata for {filename}: {e}")
            size_kb = 0
            created_str = "Unknown"

        normalized_path = os.path.abspath(os.path.normpath(path))
        file_data.append(
            {
                "name": filename,
                "size_kb": size_kb,
                "created": created_str,
                "path": normalized_path,
                "loaded": normalized_path in loaded_paths,
            }
        )

    current_data = file_data[:]
    page = 1
    per_page = 5

    ANSI_GREEN = "\033[92m"
    ANSI_RESET = "\033[0m"

    while True:
        clear_screen()
        total_pages = ceil(len(current_data) / per_page)

        # Layout settings
        name_width = 50
        size_width = 14
        created_width = 25
        buffer = 5
        total_width = name_width + size_width + created_width + 5 + buffer

        print("\n\033[1mAvailable Embedding Files:\033[0m")
        print(f"[Location: {output_folder}]")
        print("=" * total_width)
        print(
            f"{'#':<5} {'Filename':<{name_width}}  {'Size (KB)':<{size_width}}  {'Created At':<{created_width}}"
        )
        print("=" * total_width)

        start = (page - 1) * per_page
        end = start + per_page

        for i, item in enumerate(current_data[start:end], start=start + 1):
            wrapped_name = wrap(item["name"], width=name_width)
            size_str = f"{item['size_kb']:.2f} KB"
            created_str = item["created"]

            if item["loaded"]:
                print(
                    f"{ANSI_GREEN}{i:<5} {wrapped_name[0]:<{name_width}}  {size_str:<{size_width}}  {created_str:<{created_width}}{ANSI_RESET}"
                )
            else:
                print(
                    f"{i:<5} {wrapped_name[0]:<{name_width}}  {size_str:<{size_width}}  {created_str:<{created_width}}"
                )

            for line in wrapped_name[1:]:
                if item["loaded"]:
                    print(f"{ANSI_GREEN}{'':<5} {line:<{name_width}}{ANSI_RESET}")
                else:
                    print(f"{'':<5} {line:<{name_width}}")

            print()

        print("=" * total_width)
        print(f"(Page {page}/{total_pages})")

        # Options
        print("\nOptions:")
        print("  [n] Next Page")
        print("  [p] Previous Page")
        print("  [f] Filter")
        print("  [r] Reset Filters")
        print("  [s] Sort")
        print("  [l] Load Embedding File")
        print("  [q] Quit to Menu")

        cmd = input("\nEnter your choice: ").strip().lower()

        if cmd == "n":
            if page < total_pages:
                page += 1
            else:
                print("You are already on the last page.")
                input("Press Enter to continue...")
        elif cmd == "p":
            if page > 1:
                page -= 1
            else:
                print("You are already on the first page.")
                input("Press Enter to continue...")
        elif cmd == "f":
            name_filter = input("Filter by filename: ").strip().lower()
            date_filter = input("Filter by created after (YYYY-MM-DD): ").strip()
            print("\nFilter by loaded status:")
            print("  1. Only loaded files")
            print("  2. Only not-loaded files")
            print("  3. Show all (no filter)")
            status_filter = input("Choose [1/2/3]: ").strip()

            filtered = file_data
            if name_filter:
                filtered = [f for f in filtered if name_filter in f["name"].lower()]
            if date_filter:
                try:
                    dt = datetime.strptime(date_filter, "%Y-%m-%d")
                    filtered = [
                        f
                        for f in filtered
                        if f["created"] != "Unknown"
                        and datetime.strptime(f["created"], "%Y-%m-%d %H:%M:%S (UTC)")
                        >= dt
                    ]
                except ValueError:
                    print("Invalid date format. Skipping date filter.")
                    input("Press Enter to continue...")
            if status_filter == "1":
                filtered = [f for f in filtered if f["loaded"]]
            elif status_filter == "2":
                filtered = [f for f in filtered if not f["loaded"]]
            elif status_filter == "3":
                pass
            else:
                print("\033[33mInvalid status filter. Showing all files.\033[0m")

            current_data = filtered
            page = 1
        elif cmd == "r":
            current_data = file_data[:]
            page = 1
        elif cmd == "s":
            print("\nSort by:")
            print("  1. Filename")
            print("  2. Size")
            print("  3. Created At")
            sort_choice = input("Enter number: ").strip()
            sort_map = {"1": "name", "2": "size_kb", "3": "created"}
            key = sort_map.get(sort_choice)
            if key:
                reverse = input("Descending? (yes/no): ").strip().lower() == "yes"
                current_data = sorted(
                    current_data,
                    key=lambda x: x[key].lower() if isinstance(x[key], str) else x[key],
                    reverse=reverse,
                )
                page = 1
            else:
                print("Invalid sort option.")
                input("Press Enter to continue...")
        elif cmd == "l":
            selection = input("Enter the number of the file to load: ").strip()
            if selection.isdigit():
                idx = int(selection)
                if 1 <= idx <= len(current_data):
                    selected_file = current_data[idx - 1]
                    selected_path = selected_file["path"]
                    try:
                        collection = get_or_create_collection(client, collection_name)

                        # Check if the file is already loaded
                        try:
                            results = collection.get(include=["metadatas"])
                            loaded_paths = set(
                                meta.get("source_file")
                                for meta in results.get("metadatas", [])
                                if "source_file" in meta
                            )
                        except Exception as e:
                            logging.warning(
                                f"Could not retrieve loaded files from ChromaDB: {e}"
                            )
                            loaded_paths = set()

                        if selected_path in loaded_paths:
                            print(
                                f"\n\033[33mWarning:\033[0m This file is already loaded in ChromaDB."
                            )
                            confirm = (
                                input("Do you want to reload it anyway? (y/n): ")
                                .strip()
                                .lower()
                            )
                            if confirm != "y":
                                print("\033[32mSkipped reloading.\033[0m")
                                input("Press Enter to return to the menu...")
                                continue  # Return to the menu without reloading

                        # Prompt user for batch insertion
                        print("\nWould you like to use batch insertion?")
                        use_batch_input = (
                            input("Use batch insertion? (yes/no) [default: yes]: ")
                            .strip()
                            .lower()
                        )
                        use_batch = use_batch_input != "no"

                        # Ask for batch size if batch is used
                        batch_size = 100
                        if use_batch:
                            size_input = input(
                                "Enter batch size (default: 100): "
                            ).strip()
                            if size_input.isdigit():
                                batch_size = int(size_input)

                        # Load the embeddings
                        load_embeddings_into_chromadb(
                            file_path=selected_path,
                            collection=collection,
                            use_batch=use_batch,
                            batch_size=batch_size,
                        )
                        input(
                            "Embedding loaded successfully. Press Enter to continue..."
                        )
                        break
                    except Exception as e:
                        print(f"\033[31mFailed to load embeddings:\033[0m {e}")
                        input("Press Enter to continue...")
                else:
                    print("Invalid selection.")
            else:
                print("Please enter a valid number.")
        elif cmd == "q":
            break
        else:
            print("Invalid input.")
            input("Press Enter to continue...")


def view_embedding_metadata():
    """
    Displays stored metadata of embeddings in the active ChromaDB collection.
    Includes pagination, sorting, filtering, and wrapped filenames.
    """
    client = get_chromadb_client()
    config = load_config()

    if not client or not is_chromadb_active(client):
        print(
            "\033[31mError:\033[0m ChromaDB is not active. Use Option 19 or 20 first."
        )
        return

    collection_name = config.get("chromadb_collection_name", "")
    try:
        collection = get_or_create_collection(client, collection_name)
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        ids = results.get("ids", [])
    except Exception as e:
        print(f"\033[31mFailed to fetch metadata:\033[0m {e}")
        return

    if not metadatas:
        print(f"\033[33mNo metadata found in collection '{collection_name}'.\033[0m")
        return

    # Group entries by base_id prefix (removing _page_#)
    grouped = defaultdict(list)
    page_pattern = re.compile(r"^(.*)_page_\d+$")

    for i, meta in enumerate(metadatas):
        full_id = ids[i]
        match = page_pattern.match(full_id)
        base_id = match.group(1) if match else full_id  # fallback: use full_id
        grouped[base_id].append(
            {
                "custom_id": full_id,
                "model": meta.get("model", "unknown"),
                "token_count": meta.get("token_count", 0),
                "source_file": meta.get("source_file", "N/A"),
            }
        )

    # Ask user to select a group
    group_keys = sorted(grouped.keys())
    while True:
        clear_screen()
        print(
            f"\n\033[1mAvailable Document Groups in Collection: {collection_name}\033[0m"
        )
        print("=" * 70)
        for idx, key in enumerate(group_keys, start=1):
            print(f"{idx:>3}. {key}  ({len(grouped[key])} embeddings)")
        print("=" * 70)
        choice = input("\nSelect a group by number (or 'q' to quit): ").strip().lower()
        if choice == "q":
            return
        if not choice.isdigit() or int(choice) < 1 or int(choice) > len(group_keys):
            input("Invalid selection. Press Enter to try again...")
            continue

        selected_group = group_keys[int(choice) - 1]
        entries = grouped[selected_group]
        break

    total_tokens = sum(entry["token_count"] for entry in entries)
    current_data = entries[:]
    page = 1
    per_page = 10

    while True:
        clear_screen()
        total_pages = ceil(len(current_data) / per_page)

        # Table column widths
        id_width = 34
        model_width = 24
        token_width = 12
        file_width = 48
        buffer = 4
        total_width = id_width + model_width + token_width + file_width + buffer + 10

        # Header
        print(f"\n\033[1mMetadata for Group: {selected_group}\033[0m")
        print(f"Total Embeddings: {len(entries)} | Total Tokens: {total_tokens}")
        print("=" * total_width)
        print(
            f"{'#':<4} {'Custom ID':<{id_width}} {'Model':<{model_width}} {'Tokens':<{token_width}} {'Source File'}"
        )
        print("=" * total_width)

        start = (page - 1) * per_page
        end = start + per_page
        for i, entry in enumerate(current_data[start:end], start=start + 1):
            wrapped_id = wrap(entry["custom_id"], width=id_width)
            wrapped_file = wrap(entry["source_file"], width=file_width)
            model = entry["model"]
            tokens = entry["token_count"]
            max_lines = max(len(wrapped_id), len(wrapped_file))

            for line_idx in range(max_lines):
                row_id = f"{i:<4}" if line_idx == 0 else "    "
                id_str = wrapped_id[line_idx] if line_idx < len(wrapped_id) else ""
                file_str = (
                    wrapped_file[line_idx] if line_idx < len(wrapped_file) else ""
                )
                model_str = model if line_idx == 0 else ""
                token_str = str(tokens) if line_idx == 0 else ""
                print(
                    f"{row_id}{id_str:<{id_width}} {model_str:<{model_width}} {token_str:<{token_width}} {file_str}"
                )
            print()
        print("=" * total_width)
        print(f"(Page {page}/{total_pages})")

        print("\nOptions:")
        print("  [n] Next Page")
        print("  [p] Previous Page")
        print("  [f] Filter")
        print("  [r] Reset Filters")
        print("  [s] Sort")
        print("  [b] Back to Group Selection")
        print("  [q] Quit to Menu")

        cmd = input("\nEnter your choice: ").strip().lower()

        if cmd == "n":
            if page < total_pages:
                page += 1
            else:
                input("Already on the last page. Press Enter to continue...")
        elif cmd == "p":
            if page > 1:
                page -= 1
            else:
                input("Already on the first page. Press Enter to continue...")
        elif cmd == "f":
            print("\nEnter filters (leave blank to skip):")

            model_filter = input("Substring in model name: ").strip().lower()
            file_filter = input("Substring in source file name: ").strip().lower()

            token_values = [e["token_count"] for e in entries]
            min_val, max_val = min(token_values), max(token_values)
            print(f"Token count range available: {min_val}–{max_val}")
            min_tokens = input("Minimum token count: ").strip()
            max_tokens = input("Maximum token count: ").strip()

            filtered = entries
            if model_filter:
                filtered = [e for e in filtered if model_filter in e["model"].lower()]
            if file_filter:
                filtered = [
                    e for e in filtered if file_filter in e["source_file"].lower()
                ]
            if min_tokens.isdigit():
                filtered = [e for e in filtered if e["token_count"] >= int(min_tokens)]
            if max_tokens.isdigit():
                filtered = [e for e in filtered if e["token_count"] <= int(max_tokens)]

            current_data = filtered
            page = 1
        elif cmd == "r":
            current_data = entries[:]
            page = 1
        elif cmd == "s":
            print("\nSort by:")
            print("  1. Custom ID")
            print("  2. Model")
            print("  3. Token Count")
            print("  4. Source File")
            sort_choice = input("Enter number: ").strip()
            sort_map = {
                "1": "custom_id",
                "2": "model",
                "3": "token_count",
                "4": "source_file",
            }
            key = sort_map.get(sort_choice)
            if key:
                reverse = input("Descending? (yes/no): ").strip().lower() == "yes"
                current_data = sorted(
                    current_data,
                    key=lambda x: x[key].lower() if isinstance(x[key], str) else x[key],
                    reverse=reverse,
                )
                page = 1
        elif cmd == "b":
            return view_embedding_metadata()
        elif cmd == "q":
            break
        else:
            print("Invalid option.")
            input("Press Enter to continue...")


def perform_semantic_search():
    """
    Prompts the user to enter a query and performs a semantic search
    in the selected ChromaDB collection. Results are explored interactively.
    """
    client = get_chromadb_client()
    config = load_config()

    if not client or not is_chromadb_active(client):
        print(
            "\033[31mError:\033[0m ChromaDB is not active. Use Option 19 or 20 first."
        )
        return

    collection_name = config.get("chromadb_collection_name", "")
    try:
        collection = get_or_create_collection(client, collection_name)
    except Exception as e:
        print(
            f"\033[31mFailed to connect to collection '{collection_name}':\033[0m {e}"
        )
        return

    while True:
        print(
            f"\n\033[1mPerforming Semantic Search in Collection: {collection_name}\033[0m"
        )
        query = input("Enter your query: ").strip()
        if not query:
            print("No query entered. Returning to menu.")
            return

        # Optional parameters
        try:
            n_results = int(
                input("Number of top results to return [default 5]: ").strip() or 5
            )
        except ValueError:
            n_results = 5

        try:
            distance_threshold = float(
                input("Max distance threshold [optional]: ").strip() or float("inf")
            )
        except ValueError:
            distance_threshold = float("inf")

        model_override = input("Model override (leave blank to auto-detect): ").strip()

        try:
            min_tokens = int(input("Minimum token count [optional]: ").strip() or 0)
        except ValueError:
            min_tokens = 0

        try:
            max_tokens_input = input("Maximum token count [optional]: ").strip()
            max_tokens = int(max_tokens_input) if max_tokens_input else float("inf")
        except ValueError:
            max_tokens = float("inf")

        file_filter = (
            input("Filter by source filename substring [optional]: ").strip().lower()
        )

        # Detect dimensionality of the collection
        try:
            any_doc = collection.get(include=["embeddings"], limit=1)
            example_embedding = any_doc["embeddings"][0]
            collection_dim = len(example_embedding)
        except Exception as e:
            print(f"\033[31mFailed to detect collection embedding dimension:\033[0m {e}")
            return

        # Choose appropriate model
        if model_override:
            model = model_override
            dimensions = collection_dim if "text-embedding-3" in model else None
        elif collection_dim == 1536:
            model = "text-embedding-3-large"
            dimensions = 1536
        elif collection_dim == 384:
            model = "text-embedding-3-small"
            dimensions = 384
        elif collection_dim == 512:
            model = "text-embedding-ada-002"
            dimensions = None
        else:
            print(f"\033[31mUnsupported collection dimensionality: {collection_dim}\033[0m")
            return

        # Generate embedding
        try:
            print("\n\n\nGenerating query embedding...")
            response = openai.embeddings.create(
                input=query,
                model=model,
                **({"dimensions": dimensions} if dimensions else {}),
            )
            query_embedding = response.data[0].embedding
        except Exception as e:
            print(f"\033[31mFailed to generate embedding:\033[0m {e}")
            return

        # Perform the search
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["metadatas", "distances"],
            )

            ids = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            filtered_results = []
            for cid, dist, meta in zip(ids, distances, metadatas):
                if dist > distance_threshold:
                    continue
                if not (min_tokens <= meta.get("token_count", 0) <= max_tokens):
                    continue
                if file_filter and file_filter not in meta.get("source_file", "").lower():
                    continue
                similarity = 1 / (1 + dist)
                similarity_percent = similarity * 100
                filtered_results.append(
                    {
                        "custom_id": cid,
                        "distance": dist,
                        "similarity": similarity_percent,
                        "model": meta.get("model", "N/A"),
                        "token_count": meta.get("token_count", "N/A"),
                        "source_file": meta.get("source_file", "N/A"),
                    }
                )

            if not filtered_results:
                print("\033[33mNo results matched your filters.\033[0m")
                input("Press Enter to return to the menu...")
            else:
                interactive_search_results(filtered_results, query=query, collection_name=collection_name)

        except Exception as e:
            print(f"\033[31mSearch failed:\033[0m {e}")
            input("Press Enter to return to the menu...")


def interactive_search_results(filtered_results, query="", collection_name=""):
    """
    Displays semantic search results in a paginated, interactive view with filtering and sorting.
    """
    per_page = 5
    current_data = filtered_results[:]
    page = 1

    while True:
        total_pages = ceil(len(current_data) / per_page)
        start = (page - 1) * per_page
        end = start + per_page

        clear_screen()
        print(
            f"\n\033[1mSemantic Search Results for Query:\033[0m \033[94m{query} --> (Collection: {collection_name})\033[0m"
        )
        print("\n\033[1mSemantic Search Results:\033[0m")
        print("=" * 120)
        print(
            f"{'#':<4} {'Custom ID':<40} {'Distance':<10} {'Similarity %':<14} {'Tokens':<10} {'Source File'}"
        )
        print("=" * 120)

        for i, entry in enumerate(current_data[start:end], start=start + 1):
            cid = entry["custom_id"]
            dist = entry["distance"]
            sim = entry["similarity"]
            tokens = entry["token_count"]
            model = entry["model"]
            file = entry["source_file"]
            wrapped_id = wrap(cid, 40)
            wrapped_file = wrap(file, 40)
            max_lines = max(len(wrapped_id), len(wrapped_file))

            for line_idx in range(max_lines):
                row_id = f"{i:<4}" if line_idx == 0 else "    "
                id_str = wrapped_id[line_idx] if line_idx < len(wrapped_id) else ""
                file_str = (
                    wrapped_file[line_idx] if line_idx < len(wrapped_file) else ""
                )
                dist_str = f"{dist:.4f}" if line_idx == 0 else ""
                sim_str = f"{sim:.2f}%" if line_idx == 0 else ""
                token_str = str(tokens) if line_idx == 0 else ""
                print(
                    f"{row_id}{id_str:<40} {dist_str:<10} {sim_str:<14} {token_str:<10} {file_str}"
                )
            print()

        print("=" * 120)
        print(f"(Page {page}/{total_pages})")
        print("\nOptions:")
        print("  [n] Next Page")
        print("  [p] Previous Page")
        print("  [f] Filter")
        print("  [r] Reset Filters")
        print("  [s] Sort")
        print("  [q] New Query")
        print("  [x] Quit to Menu")

        cmd = input("\nEnter your choice: ").strip().lower()

        if cmd == "n":
            if page < total_pages:
                page += 1
            else:
                input("Already on last page. Press Enter...")
        elif cmd == "p":
            if page > 1:
                page -= 1
            else:
                input("Already on first page. Press Enter...")
        elif cmd == "f":
            file_sub = input("Filter by source file substring: ").strip().lower()
            model_sub = input("Filter by model substring: ").strip().lower()
            try:
                min_tokens = int(input("Min tokens: ").strip() or 0)
            except ValueError:
                min_tokens = 0
            try:
                max_tokens = int(input("Max tokens: ").strip() or 1e9)
            except ValueError:
                max_tokens = 1e9
            try:
                max_dist = float(input("Max distance: ").strip() or 1e9)
            except ValueError:
                max_dist = 1e9

            filtered = []
            for row in filtered_results:
                if file_sub and file_sub not in row["source_file"].lower():
                    continue
                if model_sub and model_sub not in row["model"].lower():
                    continue
                if not (min_tokens <= row["token_count"] <= max_tokens):
                    continue
                if row["distance"] > max_dist:
                    continue
                filtered.append(row)

            current_data = filtered
            page = 1
        elif cmd == "r":
            current_data = filtered_results[:]
            page = 1
        elif cmd == "s":
            print("\nSort by:")
            print("  1. Custom ID")
            print("  2. Distance")
            print("  3. Similarity")
            print("  4. Tokens")
            print("  5. Source File")
            key = input("Enter number: ").strip()
            key_map = {
                "1": "custom_id",
                "2": "distance",
                "3": "similarity",
                "4": "token_count",
                "5": "source_file",
            }
            sort_key = key_map.get(key)
            if sort_key:
                reverse = input("Descending? (yes/no): ").strip().lower() == "yes"
                current_data = sorted(
                    current_data,
                    key=lambda x: (
                        x[sort_key].lower()
                        if isinstance(x[sort_key], str)
                        else x[sort_key]
                    ),
                    reverse=reverse,
                )
                page = 1
        elif cmd == "q":
            return "new_query"
        elif cmd == "x":
            return "exit"
        else:
            input("Invalid option. Press Enter...")


def display_instruction_manual():
    """Displays the updated instruction manual for the OpenAI Embeddings Batch Processing Tool."""

    instruction_text = """
    OpenAI Embeddings Batch Processing Tool - Instruction Manual
    ------------------------------------------------------------

    This tool provides an interactive command-line interface for processing embeddings 
    using OpenAI’s Embeddings API (/v1/embeddings). The extracted embeddings can be 
    structured and saved for later import into vector databases such as ChromaDB.

    --------------------------------------------------------
    CONFIGURATION
    --------------------------------------------------------

    1. Set Input & Output Paths:
       - Input Folder:
         - The directory containing Markdown (.md) files, which serve as input for embedding generation.
         - Default: ./data/markdown
       - Batch Input File (.jsonl):
         - A JSON Lines (.jsonl) file is generated containing requests formatted for OpenAI’s Batch API.
         - Default: embeddings_batch_input.jsonl
       - Batch Output File (.jsonl):
         - The response file returned by OpenAI containing embeddings.
         - Default: embeddings_batch_output.jsonl
       - Output Folder:
         - Stores processed embeddings in JSON format for later use.
         - Default: ./data/vectors

    2. Select OpenAI Model:
       - The model used for embedding generation.
       - Default: text-embedding-3-small
       - Can be changed in the configuration menu.

    3. Choose Encoding Format:
       - The format in which embeddings are returned.
       - Options:
         - float (default): Standard floating-point format.
         - base64: Encoded binary representation.

    4. Set API Endpoint (if necessary):
       - The API endpoint for embedding generation.
       - Default: /v1/embeddings

    --------------------------------------------------------
    BATCH PROCESSING WORKFLOW
    --------------------------------------------------------

    5. Create Batch Input File (.jsonl)
       - Reads all Markdown (.md) files from the input folder.
       - Converts their content into JSONL format.
       - Each line in the generated .jsonl file contains:
         

{
             "custom_id": "filename_without_extension",
             "method": "POST",
             "url": "/v1/embeddings",
             "body": {
                 "model": "text-embedding-3-small",
                 "input": "Content from the markdown file",
                 "encoding_format": "float"
             }
         }


       - This file is then used for batch processing.

    6. Upload Batch Input File
       - Uploads the generated .jsonl file to OpenAI’s API.
       - Returns a **File ID**, which is required for further processing.

    7. Submit Batch Request
       - Uses the **File ID** to create a batch processing request.
       - Assigns a **Batch ID**, which tracks the job execution.
       - The batch job processes embeddings asynchronously.

    8. Monitor Batch Request
       - Checks the status of the batch processing job in real-time.
       - Possible statuses:
         - validating: The input file is being checked.
         - in_progress: The batch job is currently running.
         - finalizing: The results are being prepared.
         - completed: The batch processing is done.
         - failed: The batch job failed.
         - expired: The batch could not complete in time.
         - cancelled: The batch job was cancelled.

    9. Download Batch Results
       - Fetches the processed embeddings using the **Result File ID**.
       - Saves the raw output in embeddings_batch_output.jsonl.

    10. Save Embeddings as JSON
       - Extracts the embeddings from the batch output file.
       - Saves them in a structured JSON file named after the input folder.
       - The format ensures easy import into vector databases (e.g., ChromaDB).
       - Example JSON structure:
         

{
             "MF-194 Instruction manual for Fe/Cu-Ions generating system_page_1": {
                 "embedding": [-0.0173, 0.0430, -0.0312, ...],
                 "model": "text-embedding-3-small",
                 "tokens_used": 413
             },
             ...
         }


       - If multiple embeddings exist for the same file, they are stored under the same key.

    --------------------------------------------------------
    FILE & BATCH MANAGEMENT
    --------------------------------------------------------

    11. List Uploaded Files
       - Retrieves all uploaded files with purpose="batch".
       - Options:
         - Sort order: asc or desc (default: desc).
         - Limit: Dynamically set (must be between 1 and 10,000).
       - Displays:
         - **File ID** (can be copied to clipboard).
         - **Filename**.
         - **Size (MB)**.
         - **Upload Timestamp**.

    12. List Active Batches
       - Lists all submitted batch jobs with their status.
       - Displays:
         - **Batch ID** (can be copied to clipboard).
         - **Status** (completed, in_progress, failed, etc.).
         - **Created Timestamp**.
         - **Metadata** (if available).

    13. Cancel Batch
       - Cancels an active batch request before completion.
       - Requires confirmation before proceeding.

    --------------------------------------------------------
    API USAGE & DEBUGGING
    --------------------------------------------------------

    14. Check OpenAI API Balance
       - Fetches and displays account balance details (if accessible).
       - If restricted, provides alternative ways to check balance.

    15. View Logs
       - Displays the last N lines of the log file (embeddings_openai.log).
       - Helps in debugging API requests and responses.

    16. View This Manual
       - Displays this instruction guide.

    17. Exit Program
       - Closes the application.

    --------------------------------------------------------
    NOTES:
       - Ensure your OpenAI API key is set in the environment variable: OPENAI_API_KEY
       - Default paths:
         - Input Folder:        ./data/markdown
         - Batch Input File:    embeddings_batch_input.jsonl
         - Batch Output File:   embeddings_batch_output.jsonl
         - Output Folder:       ./data/vectors
       - Extracted embeddings are stored in a structured format for compatibility with vector databases.
       - **Use the menu options to navigate and process embedding requests efficiently.**
    """

    print(instruction_text)


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


# TODO - Implement it to menu option logic!!
def prompt_with_exit(prompt_text, valid_inputs=None, allow_empty=False):
    """
    Prompts the user for input with an option to exit.

    Args:
        prompt_text (str): The input prompt.
        valid_inputs (set or list, optional): Valid expected inputs. If None, all inputs are accepted.
        allow_empty (bool): Whether to allow empty input.

    Returns:
        str or None: The user input, or None if 'q', 'back', or 'exit' is entered.
    """
    exit_keywords = {"q", "exit", "back"}
    while True:
        user_input = input(prompt_text).strip().lower()

        if user_input in exit_keywords:
            print("\nReturning to main menu...\n")
            return None
        if not user_input and allow_empty:
            return ""
        if valid_inputs is None or user_input in valid_inputs:
            return user_input

        print(
            "Invalid input. Try again or type 'q', 'exit', or 'back' to return to the main menu."
        )


# Main Menu
def main_menu():
    """
    Main menu to interact with the batch processing workflow.
    """
    while True:
        # Default paths and parameters
        config = load_config()  # Load persisted config

        input_folder = config["input_folder"]
        batch_input_file = config["batch_input_file"]
        batch_output_file = config["batch_output_file"]
        output_folder = config["output_folder"]

        # Default parameters
        model = config["model"]
        encoding_format = config["encoding_format"]
        url = config["url"]

        # Initialize dimensionality-related settings
        dimensions = config.get("dimensions", None)
        use_dimensions = config.get("use_dimensions", False)

        # Supported models for dimensionality
        supported_dimensionality_models = {
            "text-embedding-3-small",
            "text-embedding-3-large",
        }

        # ChromaDB Configuration
        chromadb_storage_mode = config.get("chromadb_storage_mode", "persistent")
        chromadb_persistence_path = config.get(
            "chromadb_persistence_path", "./data/chromadb"
        )
        chromadb_collection_name = config.get(
            "chromadb_collection_name", "MarineEngineeringManuals"
        )
        chromadb_distance_function = config.get("chromadb_distance_function", "cosine")

        clear_screen()  # Clears terminal screen before showing the menu

        # Define ANSI color codes
        RED = "\033[31m"
        GREEN = "\033[32m"
        GREEN_BOLD = "\033[1;32m"
        RESET = "\033[0m"

        # Check ChromaDB Status Dynamically
        active_client = get_chromadb_client()
        chromadb_status = (
            "Active"
            if active_client and is_chromadb_active(active_client)
            else "Inactive"
        )

        # Apply ANSI coloring for display (but not in logic)
        status_display = (
            f"{GREEN}Active{RESET}"
            if chromadb_status == "Active"
            else f"{RED}Inactive{RESET}"
        )

        print(
            rf"""
·································································································
:  ___                      _    ___   _____           _              _     _ _                 :
: / _ \ _ __   ___ _ __    / \  |_ _| | ____|_ __ ___ | |__   ___  __| | __| (_)_ __   __ _ ___ :
:| | | | '_ \ / _ \ '_ \  / _ \  | |  |  _| | '_  _ \| '_ \ / _ \/ _ |/ _ | | '_ \ / _ / __|:
:| |_| | |_) |  __/ | | |/ ___ \ | |  | |___| | | | | | |_) |  __/ (_| | (_| | | | | | (_| \__ \:
: \___/| .__/ \___|_| |_/_/   \_\___| |_____|_| |_| |_|_.__/ \___|\__,_|\__,_|_|_| |_|\__, |___/:
:      |_|                    ____ _                               ____  ____         |___/     :
:                      _     / ___| |__  _ __ ___  _ __ ___   __ _|  _ \| __ )                  :
:                    _| |_  | |   | '_ \| '__/ _ \| '_  _ \ / _ | | | |  _ \                  :
:                   |_   _| | |___| | | | | | (_) | | | | | | (_| | |_| | |_) |                 :
:                     |_|    \____|_| |_|_|  \___/|_| |_| |_|\__,_|____/|____/      
:
                ---------------  Based on OpenAI API {GREEN_BOLD}v1.58.1{RESET}  ---------------------
            """
        )
        print(
            "\033[38;2;255;165;0m\033[1mNote:\033[0m \033[1m\033[37mDynamically changing the dimensions enables very flexible usage. For example, when using a\nvector data store that only supports embeddings up to 1024 dimensions long, developers can now\nstill use our best embedding model text-embedding-3-large and specify a value of 1024 for the\ndimensions API parameter, which will shorten the embedding down from 3072 dimensions, trading\noff some accuracy in exchange for the smaller vector size.\033[0m"
        )

        # Display the menu options

        # **General & Batch Management**
        print("\n\033[1mGeneral & Batch Management\033[0m")
        print("\t0. Terminate")
        print("\t1. View Instruction Manual")
        print("\t2. Check OpenAI API Balance")
        print("\t3. List All Batches")
        print("\t4. List Uploaded Files")
        print("\t5. Cancel Batch")
        print("\t6. View Logs")

        # **Configurations**
        print("\n\033[1mConfigurations\033[0m")
        print("\t7. Configure Paths")
        print(
            f"\t   - Input Folder: [\033[93mCurrent:\033[0m \033[1;96m{input_folder}\033[0m]"
        )
        print(
            f"\t   - Batch Input File: [\033[93mCurrent:\033[0m \033[1;96m{batch_input_file}\033[0m]"
        )
        print(
            f"\t   - Batch Output File: [\033[93mCurrent:\033[0m \033[1;96m{batch_output_file}\033[0m]"
        )
        print(
            f"\t   - Output Folder: [\033[93mCurrent:\033[0m \033[1;96m{output_folder}\033[0m]"
        )
        print(f"\t8. Change Model (\033[93mCurrent:\033[0m \033[96m{model}\033[0m)")
        print(
            f"\t9. Change Encoding Format (\033[93mCurrent:\033[0m \033[96m{encoding_format}\033[0m)"
        )
        print(f"\t10. Change API URL (\033[93mCurrent:\033[0m \033[96m{url}\033[0m)")
        # Display dimensionality option only if supported
        if model in supported_dimensionality_models:
            print(
                f"\t11. Toggle Custom Dimensionality (\033[93mCurrent:\033[0m \033[96m{'Enabled' if use_dimensions else 'Disabled'}\033[0m)"
            )
            if use_dimensions:
                print(
                    f"\t   - Current Embedding Dimensions: \033[34m{dimensions if dimensions else 'Default'}\033[0m"
                )
        else:
            print(
                "\033[2;31m\nThe dimensionality feature is only supported for text-embedding-3-small and text-embedding-3-large\n\tand later models, as per this version of OpenAI API.\033[0m"
            )

        # **Batch Processing Workflow**
        print("\n\033[1mBatch Processing Workflow\033[0m")
        print("\t12. Create Batch Input File (.jsonl)")
        print("\t13. Use Existing Batch Input File (.jsonl)")
        print("\t14. Upload Batch Input File")
        print("\t15. Create and Submit Batch Request")
        print("\t16. Monitor Batch Processing")
        print("\t17. Download Batch Results")
        print("\t18. Extract and Save Embeddings")

        # **ChromaDB - Vector Database Section**
        print(
            f"\n\033[1mChromaDB - Vector Database  │  Status: [\033[32m{status_display}\033[0m]\033[0m"
        )
        # Only show storage details if ChromaDB is active
        if chromadb_status == "Active":
            # Display storage mode dynamically
            storage_mode_label = (
                "Persistent" if chromadb_storage_mode == "persistent" else "In-Memory"
            )
            print(
                f"\t~ Storage Mode: (\033[93mCurrent:\033[0m \033[1;96m{storage_mode_label}\033[0m)"
            )

            # Show persistence path only if in persistent mode
            if chromadb_storage_mode == "persistent":
                print(
                    f"\t~ Instance Location: [\033[93mPath:\033[0m \033[1;96m{chromadb_persistence_path}\033[0m]"
                )
            print(
                f"\t~ Active Collection:  (\033[93mUsing:\033[0m \033[1;96m{chromadb_collection_name}\033[0m)\n"
            )

        print(f"\t19. Initialize ChromaDB Instance")
        print("\t20. Use Existing ChromaDB Instance or Collection")
        print("\t21. List ChromaDB Collections")
        print("\t22. Create a New ChromaDB Collection")
        print("\t23. Switch to Another Collection")
        print("\t24. Delete a ChromaDB Collection")
        print("\t25. List Available Embedding Files")
        print("\t26. Load Embeddings into ChromaDB")
        print("\t27. View Metadata of Stored Embeddings")
        print("\t28. Perform a Semantic Search")

        choice = input("\n\tEnter your choice: \t ").strip()
        clear_screen()
        try:
            # **General & Batch Management**
            if choice == "0":
                print("\nExiting the program...\n\nBye.")
                time.sleep(1)
                clear_screen()
                break
            elif choice == "1":
                # View instruction manual
                display_instruction_manual()
            elif choice == "2":
                # Check OpenAI API Balance
                get_openai_balance()
            elif choice == "3":
                # List all batches
                limit = input(
                    "Enter the number of batches to list, from newest to oldest (default: 10): \t"
                ).strip()
                limit = int(limit) if limit.isdigit() else 10
                after = (
                    input("Enter the cursor for pagination (optional): \t").strip()
                    or None
                )
                list_batches(url, limit=limit, after=after)
            elif choice == "4":
                list_uploaded_files()
            elif choice == "5":
                # Cancel a batch
                batch_id = input("Enter Batch ID to cancel: \t")
                cancel_batch(batch_id)
            elif choice == "6":
                # View log file
                view_log_file()

            # **Configurations**
            elif choice == "7":
                # Configure paths
                new_input_folder = (
                    input(f"Enter input folder (current: {input_folder}): \t").strip()
                    or input_folder
                )
                new_batch_input_file = (
                    input(
                        f"Enter batch input file (current: {batch_input_file}): \t"
                    ).strip()
                    or batch_input_file
                )
                new_batch_output_file = (
                    input(
                        f"Enter batch output file (current: {batch_output_file}): \t"
                    ).strip()
                    or batch_output_file
                )
                new_output_folder = (
                    input(f"Enter output folder (current: {output_folder}): \t").strip()
                    or output_folder
                )

                # Update values
                input_folder, batch_input_file, batch_output_file, output_folder = (
                    new_input_folder,
                    new_batch_input_file,
                    new_batch_output_file,
                    new_output_folder,
                )

                # Save new configuration
                config.update(
                    {
                        "input_folder": input_folder,
                        "batch_input_file": batch_input_file,
                        "batch_output_file": batch_output_file,
                        "output_folder": output_folder,
                    }
                )
                save_config(config)

                print(
                    f"Paths updated:\n"
                    f"Input Folder: {input_folder}\n"
                    f"Batch Input File: {batch_input_file}\n"
                    f"Batch Output File: {batch_output_file}\n"
                    f"Output Folder: {output_folder}"
                )
            elif choice == "8":
                # Change model
                new_model = input(
                    "Enter the new model (leave empty to keep current): \t"
                ).strip()
                if new_model:
                    model = new_model
                    config["model"] = model
                    save_config(config)
                print(f"\n\nModel updated to: {model}")
            elif choice == "9":
                while True:
                    print("\nSelect Encoding Format:")
                    print("1. Float (Default)")
                    print("2. Base64")

                    encoding_choice = input(
                        "Enter 1 for Float or 2 for Base64: "
                    ).strip()

                    if encoding_choice == "1":
                        new_encoding = "float"
                        break
                    elif encoding_choice == "2":
                        new_encoding = "base64"
                        break
                    else:
                        print("Invalid choice. Please enter 1 or 2.")

                config["encoding_format"] = new_encoding
                save_config(config)
                print(f"\nEncoding format updated to: {new_encoding}")
            elif choice == "10":
                # Change API URL
                new_url = input(
                    f"Enter the new API URL (Current: {url} - Press Enter to keep): \t"
                ).strip()
                if new_url:
                    url = new_url
                    config["url"] = url
                    save_config(config)
                    print(f"API URL updated to: {url}")
                else:
                    print("API URL remains unchanged.")
            elif choice == "11":
                # Toggle the use of custom dimensionality
                use_dimensions = not use_dimensions

                if use_dimensions:
                    while True:
                        print("\nSet the desired embedding dimensions.")
                        print(
                            "Note: Default is 1536 for text-embedding-3-small and 3072 for text-embedding-3-large."
                        )
                        dimensions_input = input(
                            "Enter the number of dimensions (or press Enter to keep default): "
                        ).strip()

                        if not dimensions_input:
                            dimensions = None  # Use default
                            break

                        if dimensions_input.isdigit():
                            dimensions = int(dimensions_input)

                            if model == "text-embedding-3-small" and dimensions > 1536:
                                print(
                                    "Invalid choice. text-embedding-3-small supports a maximum of 1536 dimensions."
                                )
                            elif (
                                model == "text-embedding-3-large" and dimensions > 3072
                            ):
                                print(
                                    "Invalid choice. text-embedding-3-large supports a maximum of 3072 dimensions."
                                )
                            else:
                                break
                        else:
                            print("Invalid input. Please enter a number.")
                else:
                    dimensions = None  # Reset to default

                # Save updated configuration
                config["use_dimensions"] = use_dimensions
                config["dimensions"] = dimensions
                save_config(config)

                status = "enabled" if use_dimensions else "disabled"
                print(f"\nCustom dimensionality is now {status}.")

            # **Batch Processing Workflow**
            elif choice == "12":
                # Create batch input file
                create_batch_input_jsonl(
                    input_folder,
                    batch_input_file,
                    model,
                    encoding_format,
                    dimensions,
                    use_dimensions,
                    url,
                )
            elif choice == "13":
                # Use existing batch input file
                batch_input_file = input(
                    "Enter the path of the existing .jsonl file: \t"
                ).strip()
                if not os.path.exists(batch_input_file):
                    print(f"File {batch_input_file} does not exist.")
                    logging.warning(
                        f"Attempted to use non-existent file: {batch_input_file}"
                    )
                else:
                    print(f"Using existing .jsonl file: {batch_input_file}")
                    logging.info(f"Using existing .jsonl file: {batch_input_file}")
            elif choice == "14":
                # Upload batch input file
                file_id = upload_batch_file(batch_input_file)
                print(f"Remember this File ID for later: {file_id}")
            elif choice == "15":
                # Create batch request
                file_id = input("Enter File ID: \t")
                completion_window = (
                    input("Enter completion window (default: 24h): \t").strip() or "24h"
                )
                metadata_description = input(
                    "Enter metadata description (default: 'Default batch job'): \t"
                ).strip()
                metadata = {"description": metadata_description or "Default batch job"}
                batch_id = create_batch_request(
                    file_id, completion_window=completion_window, metadata=metadata
                )
                print(f"Remember this Batch ID for later: {batch_id}")
            elif choice == "16":
                # Monitor batch processing
                batch_id = input("Enter Batch ID: \t")
                result_file_id = monitor_batch(batch_id)
                print(f"Remember this Result File ID for later: {result_file_id}")
            elif choice == "17":
                # Download batch results
                result_file_id = input("Enter Result File ID: \t")
                download_batch_results(result_file_id, batch_output_file)
            elif choice == "18":
                # Save responses as json
                save_responses_as_json(
                    batch_output_file,
                    input_folder,
                    output_folder,
                )

            # **ChromaDB Vector Database**
            elif choice == "19":
                print("\n\033[1m=== Initialize ChromaDB ===\033[0m")

                active_client = get_chromadb_client()
                is_active = active_client is not None and is_chromadb_active(
                    active_client
                )
                chromadb_status = "Active" if is_active else "Inactive"
                print(f"\nChromaDB Status: \033[1;96m{chromadb_status}\033[0m")

                if is_active:
                    print(
                        "\033[33mWarning:\033[0m A ChromaDB instance is already running."
                    )
                    confirm = (
                        input(
                            "Reinitializing may cause data loss. Do you want to proceed? (yes/no): "
                        )
                        .strip()
                        .lower()
                    )
                    if confirm != "yes":
                        print("\nChromaDB initialization aborted.")
                        continue

                while True:
                    print("\n> Choose storage mode:")
                    print("1. Persistent")
                    print("2. In-Memory")
                    storage_choice = input(
                        "Enter 1 for Persistent or 2 for In-Memory: "
                    ).strip()

                    if storage_choice == "1":
                        chromadb_storage_mode = "persistent"
                        break
                    elif storage_choice == "2":
                        chromadb_storage_mode = "in-memory"
                        break
                    else:
                        print("\033[31mInvalid choice.\033[0m Please enter 1 or 2.")

                if chromadb_storage_mode == "persistent":
                    print(f"\nCurrent persistence path: {chromadb_persistence_path}")
                    persistence_input = input("> Enter new persistence path: ").strip()
                    if persistence_input:
                        chromadb_persistence_path = persistence_input
                        if not os.path.exists(chromadb_persistence_path):
                            print(
                                "\033[31mWarning:\033[0m Path does not exist. Creating it now..."
                            )
                            os.makedirs(chromadb_persistence_path, exist_ok=True)

                new_collection_name = input(f"\n> Enter collection name: ").strip()
                if new_collection_name:
                    chromadb_collection_name = new_collection_name

                print("\n> Select Distance Function:")
                print("1. Cosine (Recommended for OpenAI Embeddings)")
                print("2. Euclidean")
                print("3. Dot Product")
                distance_choice = input("Enter 1, 2, or 3: ").strip()
                distance_functions = {
                    "1": "cosine",
                    "2": "euclidean",
                    "3": "dot_product",
                }
                chromadb_distance_function = distance_functions.get(
                    distance_choice, "cosine"
                )

                description = input(
                    "\n> Enter a short description for this collection: "
                ).strip()
                if not description:
                    description = "No description provided"
                created_at = datetime.now(timezone.utc).isoformat()

                chromadb_client = initialize_chromadb(
                    chromadb_persistence_path
                    if chromadb_storage_mode == "persistent"
                    else None
                )

                collection = chromadb_client.get_or_create_collection(
                    name=chromadb_collection_name,
                    metadata={
                        "distance_function": chromadb_distance_function,
                        "description": description,
                        "created_at": created_at,
                    },
                )

                print("\n===========================")
                print("\n\033[32mChromaDB initialized successfully.\033[0m")
                print(f"\033[90m→ Storage Mode:\033[0m {chromadb_storage_mode}")
                if chromadb_storage_mode == "persistent":
                    print(
                        f"\033[90m→ Persistence Path:\033[0m {chromadb_persistence_path}"
                    )
                print(f"\033[90m→ Collection Name:\033[0m {chromadb_collection_name}")
                print(
                    f"\033[90m→ Distance Function:\033[0m {chromadb_distance_function}"
                )
                print(f"\033[90m→ Description:\033[0m {description}")
                print(f"\033[90m→ Created At:\033[0m {created_at}")

                config["chromadb_storage_mode"] = chromadb_storage_mode
                config["chromadb_persistence_path"] = chromadb_persistence_path
                config["chromadb_collection_name"] = chromadb_collection_name
                config["chromadb_distance_function"] = chromadb_distance_function
                save_config(config)
            elif choice == "20":
                config = connect_to_existing_chromadb()
            elif choice == "21":
                config = list_chromadb_collections()
            elif choice == "22":
                config = create_new_chromadb_collection()
            elif choice == "23":
                config = switch_chromadb_collection()
            elif choice == "24":
                config = delete_chromadb_collection()
            elif choice == "25":
                output_path = config.get("output_folder", "./data/vectors")
                list_available_embedding_files(output_path)
            elif choice == "26":
                load_embeddings_into_chromadb_ui(get_chromadb_client(), load_config())
            elif choice == "27":
                view_embedding_metadata()
            elif choice == "28":
                perform_semantic_search()
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if choice != "0":
                # Pause before returning to the menu
                input("\nPress Enter to return to the menu...\t")
                clear_screen()


# Main Execution
if __name__ == "__main__":
    """
    Entry point of the program. Initializes the menu-based interface for OpenAI batch processing.
    """
    logging.info("Program started.")
    try:
        main_menu()
    except Exception as e:
        logging.critical(f"Critical error: {e}")
        raise
