import os
import json
import openai
import time
import logging
import pyperclip
import requests
from helpers import *
from dotenv import load_dotenv
import collections
import tiktoken

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
    Lists only the batches belonging to the same API as the user-defined `url` parameter with optional pagination and allows copying the full batch ID.

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
                "endpoint `/v1/dashboard/billing/credit_grants` is restricted to session keys. "
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
            # Read last `num_lines` using deque
            last_lines = collections.deque(file, num_lines)

        print("\n=== Log Output ===\n")
        for line in last_lines:
            print(line.strip())

    except Exception as e:
        print(f"An error occurred while reading the log file: {e}")


import pyperclip


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


def display_instruction_manual():
    """Displays the updated instruction manual for the OpenAI Embeddings Batch Processing Tool."""

    instruction_text = """
    OpenAI Embeddings Batch Processing Tool - Instruction Manual
    ------------------------------------------------------------

    This tool provides an interactive command-line interface for processing embeddings 
    using OpenAI’s Embeddings API (`/v1/embeddings`). The extracted embeddings can be 
    structured and saved for later import into vector databases such as ChromaDB.

    --------------------------------------------------------
    CONFIGURATION
    --------------------------------------------------------

    1. Set Input & Output Paths:
       - Input Folder:
         - The directory containing Markdown (.md) files, which serve as input for embedding generation.
         - Default: `./data/markdown`
       - Batch Input File (.jsonl):
         - A JSON Lines (.jsonl) file is generated containing requests formatted for OpenAI’s Batch API.
         - Default: `embeddings_batch_input.jsonl`
       - Batch Output File (.jsonl):
         - The response file returned by OpenAI containing embeddings.
         - Default: `embeddings_batch_output.jsonl`
       - Output Folder:
         - Stores processed embeddings in JSON format for later use.
         - Default: `./data/vectors`

    2. Select OpenAI Model:
       - The model used for embedding generation.
       - Default: `text-embedding-3-small`
       - Can be changed in the configuration menu.

    3. Choose Encoding Format:
       - The format in which embeddings are returned.
       - Options:
         - `float` (default): Standard floating-point format.
         - `base64`: Encoded binary representation.

    4. Set API Endpoint (if necessary):
       - The API endpoint for embedding generation.
       - Default: `/v1/embeddings`

    --------------------------------------------------------
    BATCH PROCESSING WORKFLOW
    --------------------------------------------------------

    5. Create Batch Input File (.jsonl)
       - Reads all Markdown (`.md`) files from the input folder.
       - Converts their content into JSONL format.
       - Each line in the generated `.jsonl` file contains:
         ```
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
         ```
       - This file is then used for batch processing.

    6. Upload Batch Input File
       - Uploads the generated `.jsonl` file to OpenAI’s API.
       - Returns a **File ID**, which is required for further processing.

    7. Submit Batch Request
       - Uses the **File ID** to create a batch processing request.
       - Assigns a **Batch ID**, which tracks the job execution.
       - The batch job processes embeddings asynchronously.

    8. Monitor Batch Request
       - Checks the status of the batch processing job in real-time.
       - Possible statuses:
         - `validating`: The input file is being checked.
         - `in_progress`: The batch job is currently running.
         - `finalizing`: The results are being prepared.
         - `completed`: The batch processing is done.
         - `failed`: The batch job failed.
         - `expired`: The batch could not complete in time.
         - `cancelled`: The batch job was cancelled.

    9. Download Batch Results
       - Fetches the processed embeddings using the **Result File ID**.
       - Saves the raw output in `embeddings_batch_output.jsonl`.

    10. Save Embeddings as JSON
       - Extracts the embeddings from the batch output file.
       - Saves them in a structured JSON file named after the input folder.
       - The format ensures easy import into vector databases (e.g., ChromaDB).
       - Example JSON structure:
         ```
         {
             "MF-194 Instruction manual for Fe/Cu-Ions generating system_page_1": {
                 "embedding": [-0.0173, 0.0430, -0.0312, ...],
                 "model": "text-embedding-3-small",
                 "tokens_used": 413
             },
             ...
         }
         ```
       - If multiple embeddings exist for the same file, they are stored under the same key.

    --------------------------------------------------------
    FILE & BATCH MANAGEMENT
    --------------------------------------------------------

    11. List Uploaded Files
       - Retrieves all uploaded files with `purpose="batch"`.
       - Options:
         - Sort order: `asc` or `desc` (default: `desc`).
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
         - **Status** (`completed`, `in_progress`, `failed`, etc.).
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
       - Displays the last `N` lines of the log file (`embeddings_openai.log`).
       - Helps in debugging API requests and responses.

    16. View This Manual
       - Displays this instruction guide.

    17. Exit Program
       - Closes the application.

    --------------------------------------------------------
    NOTES:
       - Ensure your OpenAI API key is set in the environment variable: `OPENAI_API_KEY`
       - Default paths:
         - Input Folder:        `./data/markdown`
         - Batch Input File:    `embeddings_batch_input.jsonl`
         - Batch Output File:   `embeddings_batch_output.jsonl`
         - Output Folder:       `./data/vectors`
       - Extracted embeddings are stored in a structured format for compatibility with vector databases.
       - **Use the menu options to navigate and process embedding requests efficiently.**
    """

    print(instruction_text)


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


# Main Menu
def main_menu():
    """
    Main menu to interact with the batch processing workflow.
    """
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

    while True:
        clear_screen()  # Clears terminal screen before showing the menu
        # Display the menu options
        GREEN_BOLD = "\033[1;32m"
        RESET = "\033[0m"
        print(
            rf"""
·································································································
:  ___                      _    ___   _____           _              _     _ _                 :
: / _ \ _ __   ___ _ __    / \  |_ _| | ____|_ __ ___ | |__   ___  __| | __| (_)_ __   __ _ ___ :
:| | | | '_ \ / _ \ '_ \  / _ \  | |  |  _| | '_ ` _ \| '_ \ / _ \/ _` |/ _` | | '_ \ / _` / __|:
:| |_| | |_) |  __/ | | |/ ___ \ | |  | |___| | | | | | |_) |  __/ (_| | (_| | | | | | (_| \__ \:
: \___/| .__/ \___|_| |_/_/   \_\___| |_____|_| |_| |_|_.__/ \___|\__,_|\__,_|_|_| |_|\__, |___/:
:      |_|                    ____ _                               ____  ____         |___/     :
:                      _     / ___| |__  _ __ ___  _ __ ___   __ _|  _ \| __ )                  :
:                    _| |_  | |   | '_ \| '__/ _ \| '_ ` _ \ / _` | | | |  _ \                  :
:                   |_   _| | |___| | | | | | (_) | | | | | | (_| | |_| | |_) |                 :
:                     |_|    \____|_| |_|_|  \___/|_| |_| |_|\__,_|____/|____/      
:
                ---------------  Based on OpenAI API {GREEN_BOLD}v1.58.1{RESET}  ---------------------
            """
        )
        print(
            "\033[38;2;255;165;0m\033[1mNote:\033[0m \033[1m\033[37mDynamically changing the dimensions enables very flexible usage. For example, when using a\nvector data store that only supports embeddings up to 1024 dimensions long, developers can now\nstill use our best embedding model text-embedding-3-large and specify a value of 1024 for the\ndimensions API parameter, which will shorten the embedding down from 3072 dimensions, trading\noff some accuracy in exchange for the smaller vector size.\033[0m"
        )

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
                "\033[2;31m\nThe dimensionality feature is only supported for `text-embedding-3-small` and `text-embedding-3-large`\n\tand later models, as per this version of OpenAI API.\033[0m"
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
                            "Note: Default is 1536 for `text-embedding-3-small` and 3072 for `text-embedding-3-large`."
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
                                    "Invalid choice. `text-embedding-3-small` supports a maximum of 1536 dimensions."
                                )
                            elif (
                                model == "text-embedding-3-large" and dimensions > 3072
                            ):
                                print(
                                    "Invalid choice. `text-embedding-3-large` supports a maximum of 3072 dimensions."
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
            else:
                print("Invalid choice. Please try again.")
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
