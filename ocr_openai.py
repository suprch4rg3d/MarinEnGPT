import os
import base64
import json
import openai
import time
import logging
import pyperclip
import requests
from helpers import *
import importlib.util
from dotenv import load_dotenv
from pydantic import BaseModel
import inspect
import collections

# Persistence Configuration
PERSISTENCE_FILE = "ocr_openai_persistence.json"


def load_config():
    """
    Loads the configuration file and returns stored values.
    If the file does not exist or is corrupted, it returns default values.

    Returns:
        dict: Dictionary containing stored values.
    """
    default_config = {
        "input_folder": "./data/images",
        "batch_input_file": "batch_input.jsonl",
        "batch_output_file": "batch_output.jsonl",
        "output_folder": "./data/markdown",
        "model": "gpt-4o-mini",
        "system_prompt": "You are a helpful assistant that explains instruction manuals.",
        "user_prompt": "Please interpret this image.",
        "url": "/v1/chat/completions",
        "max_tokens": 512,
        "structured_output_enabled": False,
        "schema_class_name": "MarineEngineeringManual",
        "pydantic_schema_path": "./schemas/default_schemas.py",
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

        # Initialize the schema file, ensuring it exists or can be reset
        saved_config["pydantic_schema_path"] = initialize_default_schema(
            saved_config["pydantic_schema_path"],
        )

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
    filename="ocr_openai.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)

# Load .env file if it exists
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")


# Helper Functions
def encode_image_to_base64(image_path):
    """
    Encodes an image file to a Base64 string.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64-encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def construct_request_object(
    custom_id,
    model,
    system_prompt,
    user_prompt,
    image_b64,
    max_tokens=512,
    url="/v1/chat/completions",
    structured_output_enabled=False,
    pydantic_schema_path=None,
    schema_class_name="MarineEngineeringManual",
):
    """
    Constructs a request object for the OpenAI API, optionally integrating Structured Output(See more about Structured Output API @ https://platform.openai.com/docs/guides/structured-outputs).

    Args:
        custom_id (str): Unique identifier for the request.
        model (str): Model to use for the request (e.g., gpt-4-vision).
        system_prompt (str): The system prompt.
        user_prompt (str): The user prompt.
        image_b64 (str): Base64-encoded image string.
        max_tokens (int, optional): Maximum number of tokens for the response. Defaults to 512.
        url (str, optional): API endpoint URL. Defaults to "/v1/chat/completions".
        structured_output_enabled (bool, optional): Enable Structured Output if True.
        pydantic_schema_path (str, optional): Path to the Python file containing the Pydantic schema.
        schema_class_name (str, optional): Name of the schema class in the Pydantic file.

    Returns:
        dict: Request object.
    """
    logging.info(f"Constructing request object for custom_id: {custom_id}")

    request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": url,
        "body": {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            "max_completion_tokens": max_tokens,
        },
    }

    # Structured Output
    if structured_output_enabled and pydantic_schema_path:
        try:
            logging.info("Structured Output is enabled. Attempting to load schema...")
            # Dynamically load the Pydantic schema
            spec = importlib.util.spec_from_file_location(
                "schema", pydantic_schema_path
            )
            schema_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(schema_module)

            # Load the specified schema class
            schema_class = getattr(schema_module, schema_class_name, None)
            if schema_class:
                # Generate schema for response_format
                request["body"]["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_class_name,
                        "schema": schema_class.model_json_schema(),
                    },
                }
                logging.info(
                    f"Schema '{schema_class_name}' from '{pydantic_schema_path}' was successfully loaded and applied."
                )
                print(
                    f"Structured Output integrated into request with schema '{schema_class_name}'."
                )
            else:
                logging.error(
                    f"Schema class '{schema_class_name}' not found in the specified file."
                )
                print(
                    f"Schema class '{schema_class_name}' not found. Structured Output not applied!"
                )
        except Exception as e:
            logging.error(
                f"Error loading Pydantic schema from {pydantic_schema_path}: {e}"
            )
            print(f"An error occurred while loading the schema: {e}")
    else:
        if not structured_output_enabled:
            logging.info("Structured Output is disabled. Proceeding without schema.")
        elif not pydantic_schema_path:
            logging.warning("Structured Output enabled, but no schema path provided.")

    logging.debug(f"Constructed request object: {json.dumps(request, indent=2)}")
    return request


def create_batch_input_jsonl(
    input_folder,
    output_file,
    model,
    system_prompt,
    user_prompt,
    max_tokens,
    url="/v1/chat/completions",
    structured_output_enabled=False,
    pydantic_schema_path=None,
    schema_class_name="MarineEngineeringManual",
):
    """
    Creates a .jsonl file from images in the input folder.

    Args:
        input_folder (str): Path to the folder containing images.
        output_file (str): Path to the output .jsonl file.
        model (str): Model to use for processing.
        system_prompt (str): System-level prompt for the model.
        user_prompt (str): User-level prompt for the model.
        max_tokens (int): Maximum number of tokens for the response.
        url (str, optional): API endpoint URL. Defaults to "/v1/chat/completions".
        structured_output_enabled (bool, optional): Enable Structured Output if True.
        pydantic_schema_path (str, optional): Path to the Python file containing the Pydantic schema.
        schema_class_name (str, optional): Name of the schema class in the Pydantic file.

    Raises:
        Exception: If there is an error during file creation.
    """
    logging.info("Creating batch input JSONL file...")

    try:
        with open(output_file, "w") as outfile:
            for filename in os.listdir(input_folder):
                if filename.endswith((".png", ".jpg", ".jpeg")):
                    # Process each image in the folder
                    image_path = os.path.join(input_folder, filename)
                    image_b64 = encode_image_to_base64(image_path)
                    custom_id = os.path.splitext(filename)[0]
                    # Construct and write the request object
                    request = construct_request_object(
                        custom_id=custom_id,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        image_b64=image_b64,
                        max_tokens=max_tokens,
                        url=url,
                        structured_output_enabled=structured_output_enabled,
                        pydantic_schema_path=pydantic_schema_path,
                        schema_class_name=schema_class_name,
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
    Creates a batch processing request.

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
            "description": "Default batch job for preprocessing of marine engineering manual pages as images"
        }

    try:
        response = openai.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata=metadata,
        )
        batch_id = response.id
        logging.info(f"Batch request created. Batch ID: {batch_id}")
        print(f"Batch request created. Batch ID: {batch_id}")
        return batch_id
    except Exception as e:
        logging.error(f"Error creating batch request: {e}")
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
                        f"Batch processing completed. Result File ID: {result_file_id}"
                    )
                    print(
                        f"Batch processing completed! Result File ID: {result_file_id}"
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
    Downloads the results of a completed batch request.

    Args:
        result_file_id (str): Result File ID assigned by OpenAI.
        output_file (str): Path to save the results file.

    Raises:
        Exception: If there is an error during download.
    """
    try:
        # Get file content using the result file ID
        result_file = openai.files.content(result_file_id)

        # Write the binary content to the output file
        with open(output_file, "wb") as f:
            f.write(result_file.read())

        logging.info(f"Results downloaded: {output_file}")
        print(f"Results downloaded: {output_file}")
    except Exception as e:
        logging.error(f"Error downloading batch results: {e}")
        raise


def load_schema_fields(
    pydantic_schema_path,
    schema_class_name,
):
    """
    Loads schema fields dynamically from a specified Pydantic schema file.

    Args:
        pydantic_schema_path (str): Path to the Python file containing the Pydantic schema.
        schema_class_name (str): Name of the schema class in the Pydantic file.

    Returns:
        list: A list of schema field names if successful, otherwise an empty list.
    """
    if not pydantic_schema_path or not os.path.isfile(pydantic_schema_path):
        logging.warning(
            f"[WARNING] Pydantic schema file not found at {pydantic_schema_path}"
        )
        print(f"[DEBUG] Schema file not found: {pydantic_schema_path}")
        return []

    try:
        print(f"[DEBUG] Loading schema from: {pydantic_schema_path}")
        logging.info(f"[INFO] Loading schema from {pydantic_schema_path}")

        # Load the schema dynamically
        spec = importlib.util.spec_from_file_location("schema", pydantic_schema_path)
        schema_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(schema_module)

        logging.info("[INFO] Schema module loaded successfully.")
        print(f"[DEBUG] Schema module loaded successfully.")

        # Retrieve the schema class
        schema_class = getattr(schema_module, schema_class_name, None)
        if not schema_class:
            logging.error(
                f"Schema class '{schema_class_name}' not found in {pydantic_schema_path}"
            )
            print(f"[DEBUG] Schema class not found: {schema_class_name}")
            return []

        logging.info(f"[INFO] Found schema class: {schema_class_name}")
        print(f"[DEBUG] Found schema class: {schema_class_name}")

        # Extract field names from Pydantic model
        schema_fields = (
            list(schema_class.model_fields.keys())
            if hasattr(schema_class, "model_fields")
            else []
        )

        logging.debug(f"[DEBUG] Extracted schema fields: {schema_fields}")
        print(f"[DEBUG] Extracted fields: {schema_fields}")

        if not schema_fields:
            logging.warning(
                f"[WARNING] No fields found in schema class {schema_class_name}"
            )
            print(f"[DEBUG] No fields found in schema class {schema_class_name}")

        return schema_fields

    except Exception as e:
        logging.error(f"[ERROR] Error loading schema from {pydantic_schema_path}: {e}")
        print(f"[DEBUG] Error loading schema: {e}")
        return []


def save_responses_as_markdown(
    result_file,
    input_folder,
    output_folder="./data/markdown",
    pydantic_schema_path=None,
    schema_class_name="MarineEngineeringManual",
):
    """
    Saves responses from the results file as Markdown files in a subfolder based on the base name of the input folder.
    Dynamically adapts to the schema fields provided.

    Args:
        result_file (str): Path to the results file.
        input_folder (str): Path to the input folder containing the images.
        output_folder (str): Base folder for saving Markdown files.
        pydantic_schema_path (str, optional): Path to the Pydantic schema file. Defaults to None.
        schema_class_name (str, optional): Name of the schema class in the Pydantic file. Defaults to "MarineEngineeringManual".

    Raises:
        Exception: If there is an error during file processing.
    """
    try:
        # Create a subfolder in the output folder based on the base name of the input folder
        base_name = os.path.basename(os.path.normpath(input_folder))
        markdown_folder = os.path.join(output_folder, base_name)
        os.makedirs(markdown_folder, exist_ok=True)  # Ensure the folder exists

        print(f"[DEBUG] Using schema path: {pydantic_schema_path}")
        print(f"[DEBUG] Using schema class: {schema_class_name}")

        # Load schema fields dynamically
        schema_fields = load_schema_fields(pydantic_schema_path, schema_class_name)

        # Debugging: Print schema fields
        print(f"[DEBUG] Schema Fields: {schema_fields}")

        with open(result_file, "r", encoding="utf-8") as infile:
            for line in infile:
                try:
                    # Parse each response line
                    response = json.loads(line)
                    custom_id = response.get("custom_id", "unknown_id")

                    if response.get("response", {}).get("status_code") == 200:
                        # Extract message content
                        choices = response["response"]["body"].get("choices", [])
                        if not choices:
                            logging.warning(
                                f"No choices found in response for {custom_id}"
                            )
                            continue

                        message = choices[0].get("message", {})
                        finish_reason = choices[0].get("finish_reason", "")
                        refusal = message.get("refusal", None)
                        message_content = message.get("content", "")

                        markdown_content = f"# Response for {custom_id}\n\n"

                        # Handle refusals (i.e."refusal": "I'm sorry, I cannot assist with that request.")
                        if refusal:
                            logging.warning(
                                f"Skipping Markdown generation for {custom_id} due to model refusal: {refusal}"
                            )
                            continue  # Skip this response entirely

                        # Handle truncated responses
                        elif finish_reason == "length":
                            logging.warning(
                                f"Truncated response detected for {custom_id}. Consider increasing max_tokens."
                            )
                            markdown_content += "## WARNING: Response was truncated.\nSome structured data may be missing due to max token limits.\n\n"

                        # Attempt to parse structured output
                        try:
                            parsed_content = json.loads(message_content)

                            # Generate Markdown from structured fields
                            if schema_fields:
                                for field in schema_fields:
                                    value = parsed_content.get(field, "N/A")
                                    markdown_content += f"## {field.replace('_', ' ').title()}\n{value}\n\n"

                            else:
                                logging.warning(
                                    f"No structured output detected for {custom_id}. Check your schema settings."
                                )
                                markdown_content += "## No structured output detected. Check your schema settings.\n"

                        except json.JSONDecodeError:
                            logging.error(
                                f"JSON parsing failed for {custom_id}. Fallback to raw text."
                            )
                            markdown_content += (
                                f"## Assistant Response\n{message_content}\n"
                            )

                        # Save the Markdown file
                        output_md_path = os.path.join(
                            markdown_folder, f"{custom_id}.md"
                        )
                        with open(output_md_path, "w", encoding="utf-8") as outfile:
                            outfile.write(markdown_content.strip())

                        logging.info(f"Saved Markdown file: {output_md_path}")
                        print(f"Saved Markdown: {output_md_path}")
                    else:
                        logging.warning(
                            f"Skipping response for custom_id '{custom_id}' due to non-200 status code."
                        )
                except Exception as parse_error:
                    logging.error(
                        f"Error parsing response line: {line}. Error: {parse_error}"
                    )
                    print(f"Error parsing a response. See logs for details.")
    except Exception as e:
        logging.error(f"Error saving responses as Markdown: {e}")
        raise


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


def list_batches(url, limit=10, after=None):
    """
    Lists only the batches belonging to the same API as the user-defined `url` parameter with optional pagination and allows copying the full batch ID.

    Args:
        url (str): The current OpenAI API endpoint (e.g., "/v1/chat/completions").
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
        print(f"\n=== Batches for \033[96m{url}\033[0m ===")
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
        logging.error(f"Error listing batches: {e}")
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


def initialize_default_schema(persisted_path=None):
    """
    Ensures the schemas directory and default_schemas.py file exist.
    If persisted_path exists, it is returned. Otherwise, the default path is used.

    Args:
        persisted_path (str): Path from the persistence file, if available.

    Returns:
        str: The final schema file path.
    """
    schema_dir = "./schemas"
    default_schema_file = os.path.join(schema_dir, "default_schemas.py")

    # If a persisted schema path exists, return it
    if persisted_path and os.path.isfile(persisted_path):
        return persisted_path

    # Create the directory if it doesn't exist
    if not os.path.exists(schema_dir):
        os.makedirs(schema_dir)
        logging.info(f"Created schema directory: {schema_dir}")
        print(f"Schema directory created at {schema_dir}.")

    # Check if the schema file exists and prompt the user for reset
    if os.path.isfile(default_schema_file):
        print(f"\nDefault schema file already exists at {default_schema_file}.")
        choice = (
            input(
                "Do you want to reset the default_schemas.py file? "
                "Any custom changes will be lost (yes/no): \t"
            )
            .strip()
            .lower()
        )

        if choice not in {"yes", "y"}:
            logging.info(f"Retained existing schema file: {default_schema_file}")
            return default_schema_file  # Keep the existing file

        action = "reset"
    else:
        action = "created"

    # Write the default schema file (either creating or resetting)
    with open(default_schema_file, "w", encoding="utf-8") as f:
        f.write(
            """
from pydantic import BaseModel, Field
from typing import Optional, List

class Schematic(BaseModel):
    \"\"\"Schema for representing schematic or diagram metadata.\"\"\"
    description: str = Field(..., description="Description of the schematic.")
    annotations: Optional[List[str]] = Field(
        None, description="Notes or labels associated with the schematic."
    )
    schematic_relative_path: Optional[str] = Field(
        None, description="Relative path to the schematic file/image."
    )
    schematic_ocr_text: Optional[str] = Field(
        None, description="OCR text extracted from the schematic, if any."
    )

    class Config:
        json_schema_extra = {"additionalProperties": False}


class MarineEngineeringManual(BaseModel):
    \"\"\"Pydantic schema for structuring marine engineering manual data.\"\"\"
    image_relative_path: str = Field(
        ..., description="Relative path of the manual image."
    )
    ocr_text: str = Field(..., description="OCR output from the image.")
    ocr_explanation: str = Field(..., description="Explanation of the OCR text.")
    manual_title: Optional[str] = Field(
        None, description="Title of the manual."
    )
    chapter_name: Optional[str] = Field(
        None, description="Chapter name or section."
    )
    diagrams: Optional[List[str]] = Field(
        None, description="Descriptions of diagrams in the manual."
    )
    schematics: Optional[List[Schematic]] = Field(
        None, description="List of schematic metadata."
    )
    additional_notes: Optional[str] = Field(
        None, description="Additional notes about the manual page."
    )

    class Config:
        json_schema_extra = {"additionalProperties": False}

"""
        )

    # Log the appropriate action
    if action == "created":
        logging.info(f"Default schema file created: {default_schema_file}")
        print(f"Default schema file successfully created at {default_schema_file}.")
    elif action == "reset":
        logging.info(f"Default schema file reset: {default_schema_file}")
        print(f"Default schema file successfully reset at {default_schema_file}.")

    return default_schema_file


def list_pydantic_classes(pydantic_schema_path):
    """
    Lists all Pydantic classes from a given schema file.

    Args:
        pydantic_schema_path (str): Path to the Python file containing the Pydantic schema.

    Returns:
        list: A list of Pydantic class names if found, otherwise an empty list.
    """
    if not pydantic_schema_path or not os.path.isfile(pydantic_schema_path):
        logging.warning(f"Pydantic schema file not found: {pydantic_schema_path}")
        print(f"Error: Schema file '{pydantic_schema_path}' not found.")
        return []

    try:
        # Dynamically load the schema module
        spec = importlib.util.spec_from_file_location(
            "schema_module", pydantic_schema_path
        )
        schema_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(schema_module)

        # Retrieve all classes from the module that inherit from BaseModel
        pydantic_classes = [
            name
            for name, obj in inspect.getmembers(schema_module, inspect.isclass)
            if issubclass(obj, BaseModel) and obj.__module__ == schema_module.__name__
        ]

        if not pydantic_classes:
            logging.warning(f"No Pydantic classes found in: {pydantic_schema_path}")
            print(f"No Pydantic classes found in the schema file.")

        return pydantic_classes

    except Exception as e:
        logging.error(
            f"Error loading Pydantic classes from {pydantic_schema_path}: {e}"
        )
        print(f"Error: Could not load schema classes due to {e}")
        return []


def view_log_file(log_file="ocr_openai.log"):
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
            # Efficiently read last `num_lines` using deque
            last_lines = collections.deque(file, num_lines)

        print("\n=== Log Output ===\n")
        for line in last_lines:
            print(line.strip())

    except Exception as e:
        print(f"An error occurred while reading the log file: {e}")


def display_instruction_manual():
    """
    Displays a detailed instruction manual for the user.
    """
    manual = """
=== OpenAI Batch Processing Tool ===

This tool provides an interactive menu to handle batch processing workflows with OpenAI's API.

=== Features and Instructions ===

1. Configure Paths
   - Update the paths for:
     - Input folder (where your images are stored).
     - Batch input file (.jsonl file containing requests).
     - Batch output file (location to save processed results).
     - Output folder (to save Markdown files for processed results).

2. Set max_tokens
   - Set the maximum number of tokens for responses generated by the model (default: 512).

3. Create Batch Input File (.jsonl)
   - Generate a `.jsonl` file from images in the input folder.
   - Each request will include system and user prompts, the image, and settings for model processing.

4. Use Existing Batch Input File (.jsonl)
   - Reuse a pre-existing `.jsonl` file without creating a new one.

5. Upload Batch Input File
   - Upload a `.jsonl` file to OpenAI for batch processing.
   - You will receive a File ID for the uploaded file.

6. Create and Submit Batch Request
   - Use the File ID to submit a batch processing job.
   - Specify:
     - Completion window (default: 24h).
     - Metadata description (default: "Default batch job").
   - You will receive a Batch ID for the submitted job.

7. Monitor Batch Processing
   - Enter the Batch ID to check the job's current status.
   - Supported statuses:
     - `validating`: Input file is being validated.
     - `in_progress`: Batch is being processed.
     - `finalizing`: Results are being prepared.
     - `completed`: Results are ready.
     - `expired`: Batch was not completed within the specified window.
     - `cancelling`: Batch is being cancelled.
     - `cancelled`: Batch was cancelled.
   - If the batch is completed, you will get the Result File ID.

8. Download Batch Results
   - Enter the Result File ID to download the results file.
   - The results will be saved to the specified batch output file.

9. Save Responses as Markdown
   - Convert responses from the results file into Markdown files.
   - Each file will include the content from the response for better readability.

10. Change API URL
    - Update the OpenAI API endpoint URL. 
    - Default: `/v1/chat/completions`.

11. Change Model
    - Update the model being used for processing requests (e.g., `gpt-4`, `gpt-4-vision`).

12. Enable/Disable Structured Output
    - Toggle the Structured Output feature on or off.
    - Structured Outputs enable the model to adhere to a specified JSON schema.

13. Set Pydantic Schema Path
    - Specify the path to the Python file containing the Pydantic schema.
    - The schema ensures structured responses from the OpenAI API.

14. Default Schema Initialization
    - The tool initializes a default Pydantic schema (`MarineEngineeringManual`) in `schemas/default_schemas.py`.
    - This schema is used for structuring manual page data (e.g., OCR text, schematics metadata).

15. Check OpenAI API Balance
    - Retrieves and displays the current OpenAI API balance or explains any restrictions.

16. View Instruction Manual
    - Displays this guide.

17. Cancel Batch
    - Enter the Batch ID to cancel an ongoing batch.
    - Confirm before proceeding with cancellation.
    - It may take up to 10 minutes for the batch status to change to `cancelled`.

18. List All Batches
    - View batches with optional pagination:
      - Specify the number of batches to list (default: 10).
      - Use the cursor for pagination to fetch additional results.
    - Displays:
      - Batch ID (truncated for readability).
      - Status (e.g., `completed`, `in_progress`).
      - Created time (formatted for clarity).
      - Metadata description (if available).
    - Option to copy the full Batch ID to your clipboard for further use.

=== Notes ===

- **Structured Output Feature Documentation for Python:**
  1. **Using Pydantic for Structured Outputs**:
     - Define the schema as a Python class inheriting from `BaseModel`.
     - Use descriptive fields with type annotations and `Field` metadata.
     - Ensure compliance with OpenAI Structured Outputs requirements:
       - `additionalProperties` must always be set to `false`.
       - Objects may have up to 100 properties and 5 levels of nesting.
       - Use enums judiciously, adhering to character limits.

     Example Schema:
     ```python
     from pydantic import BaseModel, Field
     from typing import Optional, List

     class ExampleSchema(BaseModel):
         field1: str = Field(..., description="Description for field1.")
         field2: Optional[int] = Field(None, description="Optional integer field.")
         items: List[str] = Field(..., description="List of strings.")

         class Config:
             schema_extra = {"additionalProperties": False}
     ```

  2. **Integration with OpenAI API**:
     - Use the schema in the `construct_request_object()` function.
     - Dynamically load the schema at runtime from the configured path.
     - Toggle the feature on/off from the menu.

  3. **Tweaking the Default Schema**:
     - The default schema is located at `schemas/default_schemas.py`.
     - Use the `MarineEngineeringManual` schema to structure manual data:
       - OCR text and its explanation.
       - Metadata about schematics and diagrams.

  4. **Error Handling**:
     - The tool validates the schema path and logs errors if the schema cannot be loaded or applied.

- **Ensure your OpenAI API key is set in the environment variable `OPENAI_API_KEY`.
- Detailed logs are stored in `ocr_openai.log`.

===============================
"""
    print(manual)
    logging.info("Displayed the instruction manual.")


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
    pydantic_schema_path = config[
        "pydantic_schema_path"
    ]  # Ensures default schema setup

    structured_output_enabled = config[
        "structured_output_enabled"
    ]  # Flag for enabling structured output
    schema_class_name = config["schema_class_name"]

    # Default parameters
    model = config["model"]
    system_prompt = config["system_prompt"]
    user_prompt = config["user_prompt"]
    url = config["url"]
    max_tokens = config["max_tokens"]  # Default max_tokens value

    while True:
        clear_screen()  # Clears terminal screen before showing the menu
        # Display the menu options
        GREEN_BOLD = "\033[1;32m"
        RESET = "\033[0m"
        print(
            rf"""
······················································································
:  ___                      _    ___    ___   ____ ____    ____        _       _     :
: / _ \ _ __   ___ _ __    / \  |_ _|  / _ \ / ___|  _ \  | __ )  __ _| |_ ___| |__  :
:| | | | '_ \ / _ \ '_ \  / _ \  | |  | | | | |   | |_) | |  _ \ / _` | __/ __| '_ \ :
:| |_| | |_) |  __/ | | |/ ___ \ | |  | |_| | |___|  _ <  | |_) | (_| | || (__| | | |:
: \___/| .__/ \___|_| |_/_/   \_\___|  \___/ \____|_| \_\ |____/ \__,_|\__\___|_| |_|:
:      |_|      |  _ \ _ __ ___   ___ ___  ___ ___(_)_ __   __ _                     :
:               | |_) | '__/ _ \ / __/ _ \/ __/ __| | '_ \ / _` |                    :
:               |  __/| | | (_) | (_|  __/\__ \__ \ | | | | (_| |                    :
:               |_|   |_|  \___/ \___\___||___/___/_|_| |_|\__, |                    :
:                                                          |___/                     :
        ---------------  Based on OpenAI API {GREEN_BOLD}v1.58.1{RESET}  ---------------------            
                        
            """
        )
        print(
            "\033[38;2;255;165;0m\033[1mNote:\033[0m \033[1m\033[4mBy default, the latest models are limited to 4,096 output tokens independent of the context window size!\033[0m"
        )

        # **General & Batch Management**
        print("\n\033[1mGeneral & Batch Management\033[0m")
        print("\t0. Terminate")
        print("\t1. View Instruction Manual")
        print("\t2. Check OpenAI API Balance")
        print("\t3. List All Batches")
        print("\t4. Cancel Batch")
        print("\t5. View Logs")

        # **Configurations**
        print("\n\033[1mConfigurations\033[0m")
        print("\t6. Configure Paths")
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
        print(f"\t7. Change Model (\033[93mCurrent:\033[0m \033[96m{model}\033[0m)")
        print(
            f"\t8. Set Max Tokens (\033[93mCurrent:\033[0m \033[96m{max_tokens}\033[0m)"
        )
        print(f"\t9. Change API URL (\033[93mCurrent:\033[0m \033[96m{url}\033[0m)")
        print("\t10. Change & View System/User Prompts")
        print(
            f"\t11. Enable/Disable Structured Output (\033[93mCurrent:\033[0m \033[96m{'Enabled' if structured_output_enabled else 'Disabled'}\033[0m)"
        )
        print(
            f"\t12. Set Pydantic Schema Path (\033[93mCurrent:\033[0m \033[96m{pydantic_schema_path or 'Not Set'}\033[0m)"
        )
        print(
            f"\t13. Change Schema Class Name (\033[93mCurrent:\033[0m \033[96m{schema_class_name}\033[0m)"
        )

        # **Batch Processing Workflow**
        print("\n\033[1mBatch Processing Workflow\033[0m")
        print("\t14. Create Batch Input File (.jsonl)")
        print("\t15. Use Existing Batch Input File (.jsonl)")
        print("\t16. Upload Batch Input File")
        print("\t17. Create and Submit Batch Request")
        print("\t18. Monitor Batch Processing")
        print("\t19. Download Batch Results")
        print("\t20. Save Responses as Markdown")

        choice = input("\n\tEnter your choice: \t ").strip()
        clear_screen()
        try:
            # **General & Batch Management**
            if choice == "0":
                print("\nExiting the program...\n\nBye.")
                time.sleep(2)
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
                # Cancel a batch
                batch_id = input("Enter Batch ID to cancel: \t")
                cancel_batch(batch_id)
            elif choice == "5":
                # View log file
                view_log_file()

            # **Configurations**
            elif choice == "6":
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
            elif choice == "7":
                # Change model
                new_model = input(
                    "Enter the new model (leave empty to reset to default): \t"
                ).strip()
                if new_model:
                    model = new_model
                    config["model"] = model
                    save_config(config)
                print(f"\n\nModel updated to: {model}")
            elif choice == "8":
                # Update max_tokens
                max_tokens_input = input(
                    "Enter maximum tokens for responses (current: {}): \t".format(
                        max_tokens
                    )
                ).strip()
                if max_tokens_input.isdigit():
                    max_tokens = int(max_tokens_input)
                    config["max_tokens"] = max_tokens
                    save_config(config)
                    print(f"`Maximum completion tokens were updated to {max_tokens}")
                else:
                    print("Invalid input. Please enter a positive integer.")
            elif choice == "9":
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
            elif choice == "10":
                print("\n=== Current Prompts ===")
                print(f"\n[System Prompt]:\n{system_prompt}\n")
                print(f"[User Prompt]:\n{user_prompt}\n")

                # Prompt user for new system prompt
                new_system_prompt = input(
                    "Enter new System Prompt (or press Enter to keep current): \t"
                ).strip()
                if new_system_prompt:
                    system_prompt = new_system_prompt
                    config["system_prompt"] = system_prompt
                    logging.info("Updated system prompt.")

                # Prompt user for new user prompt
                new_user_prompt = input(
                    "Enter new User Prompt (or press Enter to keep current): \t"
                ).strip()
                if new_user_prompt:
                    config["user_prompt"] = user_prompt
                    logging.info("Updated user prompt.")

                save_config(config)
                print("\nPrompts updated successfully.\n")
            elif choice == "11":
                # Toggle Structured Output
                structured_output_enabled = not structured_output_enabled
                config["structured_output_enabled"] = structured_output_enabled
                save_config(config)
                state = "enabled" if structured_output_enabled else "disabled"
                logging.info(f"Structured Output feature has been {state}.")
                print(f"Structured Output is now {state}.")
            elif choice == "12":
                # Set Pydantic Schema Path
                new_path = input(
                    f"Enter the new Pydantic Schema Path to file (Current: {pydantic_schema_path} - Press Enter to keep): \t"
                ).strip()

                if new_path:
                    if os.path.isfile(new_path):
                        pydantic_schema_path = new_path
                        config["pydantic_schema_path"] = pydantic_schema_path
                        save_config(config)
                        print(
                            f"Pydantic schema path updated to: {pydantic_schema_path}"
                        )
                    else:
                        logging.warning(f"Invalid schema path provided: {new_path}")
                        print(
                            f"Error: The file {new_path} does not exist. Keeping the current path."
                        )
                else:
                    print("No changes made to the Pydantic schema path.")
            elif choice == "13":  # Assign a new menu option for schema selection
                print(
                    f"\n=== Available Pydantic Schemas in {pydantic_schema_path}===\n"
                )
                available_schemas = list_pydantic_classes(pydantic_schema_path)

                if not available_schemas:
                    print(
                        "\nNo available schemas found. Ensure the schema file is correctly defined."
                    )
                else:
                    for idx, schema in enumerate(available_schemas, start=1):
                        print(f"{idx}. {schema}")

                    schema_choice = input(
                        "\nEnter the number corresponding to your desired schema (press Enter to keep current): \t"
                    ).strip()

                    if not schema_choice:  # User pressed Enter, keep current schema
                        print(f"\n\nSchema class remains: {schema_class_name}")
                    elif schema_choice.isdigit():
                        schema_choice = int(schema_choice)
                        if 1 <= schema_choice <= len(available_schemas):
                            schema_class_name = available_schemas[schema_choice - 1]
                            config["schema_class_name"] = schema_class_name
                            save_config(config)
                            logging.info(
                                f"Schema class updated to: {schema_class_name}"
                            )
                            print(f"\n\nSchema class updated to: {schema_class_name}")
                        else:
                            print("\nInvalid choice. Please enter a valid number.")
                    else:
                        print(
                            "\nInvalid input. Please enter a numeric value or press Enter to skip."
                        )

            # **Batch Processing Workflow**
            elif choice == "14":
                # Create batch input file
                create_batch_input_jsonl(
                    input_folder,
                    batch_input_file,
                    model,
                    system_prompt,
                    user_prompt,
                    max_tokens,
                    url,
                    structured_output_enabled,
                    pydantic_schema_path,
                )
            elif choice == "15":
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
            elif choice == "16":
                # Upload batch input file
                file_id = upload_batch_file(batch_input_file)
                print(f"Remember this File ID for later: {file_id}")
            elif choice == "17":
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
            elif choice == "18":
                # Monitor batch processing
                batch_id = input("Enter Batch ID: \t")
                result_file_id = monitor_batch(batch_id)
                print(f"Remember this Result File ID for later: {result_file_id}")
            elif choice == "19":
                # Download batch results
                result_file_id = input("Enter Result File ID: \t")
                download_batch_results(result_file_id, batch_output_file)
            elif choice == "20":
                # Save responses as Markdown
                save_responses_as_markdown(
                    batch_output_file,
                    input_folder,
                    output_folder,
                    pydantic_schema_path,
                    schema_class_name,
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
