import os
import base64
import json
import openai
import time
import logging
import pyperclip
import requests
from helpers import *

# Logging Configuration
logging.basicConfig(
    filename="ocr_openai.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)

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
):
    """
    Constructs a request object for the OpenAI API.

    Args:
        custom_id (str): Unique identifier for the request.
        model (str): Model to use for the request (e.g., gpt-4-vision).
        system_prompt (str): The system prompt.
        user_prompt (str): The user prompt.
        image_b64 (str): Base64-encoded image string.
        max_tokens (int, optional): Maximum number of tokens for the response. Defaults to 512.
        url (str, optional): API endpoint URL. Defaults to "/v1/chat/completions".

    Returns:
        dict: Request object.
    """
    return {
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
            "max_tokens": max_tokens,
        },
    }


def create_batch_input_jsonl(
    input_folder,
    output_file,
    model,
    system_prompt,
    user_prompt,
    max_tokens,
    url="/v1/chat/completions",
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

    Raises:
        Exception: If there is an error during file creation.
    """
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


def create_batch_request(file_id, completion_window="24h", metadata=None):
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
        metadata = {"description": "Default batch job"}

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


def save_responses_as_markdown(result_file, output_folder):
    """
    Saves responses from the results file as Markdown files.

    Args:
        result_file (str): Path to the results file.
        output_folder (str): Path to the folder where Markdown files will be saved.

    Raises:
        Exception: If there is an error during file processing.
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        with open(result_file, "r") as infile:
            for line in infile:
                # Parse each response line
                response = json.loads(line)
                custom_id = response["custom_id"]
                explanation = response["body"]["choices"][0]["message"]["content"]
                output_md_path = os.path.join(output_folder, f"{custom_id}.md")
                # Save the explanation as a Markdown file
                with open(output_md_path, "w") as outfile:
                    outfile.write(f"# Page Interpretation\n\n{explanation}")
                logging.info(f"Saved Markdown file: {output_md_path}")
                print(f"Saved Markdown: {output_md_path}")
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
        input(f"Are you sure you want to cancel Batch ID {batch_id}? (yes/no): ")
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


def list_batches(limit=10, after=None):
    """
    Lists batches with optional pagination and allows copying the full batch ID.

    Args:
        limit (int): Number of batches to list per page (default is 10).
        after (str, optional): Cursor for pagination to fetch results after a specific batch.

    Returns:
        None
    """
    try:
        params = {"limit": limit}
        if after:
            params["after"] = after

        # Retrieve batches with pagination
        response = openai.batches.list(**params)

        # Store batch IDs for clipboard functionality
        batch_ids = {}

        # Display header
        print("\n=== Batches ===")
        print(
            f"{'Index':<6} {'Batch ID (truncated)':<20} {'Status':<15} {'Created At':<20} {'Metadata':<30}"
        )
        print("-" * 100)

        # Display only the limited number of batches
        for i, batch in enumerate(response, start=1):
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
            "Enter the index of the batch to copy its full ID (or press Enter to skip): "
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
            "Content-Type": "application/json"
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
            "total_available": total_available
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

12. View Instruction Manual
    - Displays this guide.

13. Cancel Batch
    - Enter the Batch ID to cancel an ongoing batch.
    - Confirm before proceeding with cancellation.
    - It may take up to 10 minutes for the batch status to change to `cancelled`.

14. List All Batches
    - View batches with optional pagination:
      - Specify the number of batches to list (default: 10).
      - Use the cursor for pagination to fetch additional results.
    - Displays:
      - Batch ID (truncated for readability).
      - Status (e.g., `completed`, `in_progress`).
      - Created time (formatted for clarity).
      - Metadata description (if available).
    - Option to copy the full Batch ID to your clipboard for further use.

0. Terminate
   - Exit the program gracefully.

=== Notes ===
- Ensure your OpenAI API key is set in the environment variable `OPENAI_API_KEY`.
- You can use `pyperclip` to copy Batch IDs or other information as needed.
- If any errors occur, detailed logs are stored in `ocr_openai.log`.

===============================
"""
    print(manual)
    logging.info("Displayed the updated instruction manual.")


# Main Menu
def main_menu():
    """
    Main menu to interact with the batch processing workflow.
    """
    # Default paths and parameters
    input_folder = "./data/images"
    batch_input_file = "batch_input.jsonl"
    batch_output_file = "batch_output.jsonl"
    output_folder = "./data/markdown"

    # Default parameters
    model = "gpt-4"
    system_prompt = "You are a helpful assistant that explains instruction manuals."
    user_prompt = "Please interpret this image."
    url = "/v1/chat/completions"
    max_tokens = 512  # Default max_tokens value

    while True:
        # Display the menu options
        print(
            r"""
___  ___              _         _____        _____ ______  _____                                    
|  \/  |             (_)       |  ___|      |  __ \| ___ \|_   _|                                   
| .  . |  __ _  _ __  _  _ __  | |__  _ __  | |  \/| |_/ /  | |                                     
| |\/| | / _` || '__|| || '_ \ |  __|| '_ \ | | __ |  __/   | |                                     
| |  | || (_| || |   | || | | || |___| | | || |_\ \| |      | |                                     
\______/ \__,_||_|   |_||_| |__________| |____________ _______/     ______         _         _      
|  _  |                     / _ \|_   _| |  _  |/  __ \| ___ \  _   | ___ \       | |       | |     
| | | | _ __    ___  _ __  / /_\ \ | |   | | | || /  \/| |_/ /_| |_ | |_/ /  __ _ | |_  ___ | |__   
| | | || '_ \  / _ \| '_ \ |  _  | | |   | | | || |    |    /|_   _|| ___ \ / _` || __|/ __|| '_ \  
\ \_/ /| |_) ||  __/| | | || | | |_| |_  \ \_/ /| \__/\| |\ \  |_|  | |_/ /| (_| || |_| (__ | | | | 
 \___/ | .__/  \___||_| |_|\_| |_/\___/   \___/  \____/\_| \_|      \____/  \__,_| \__|\___||_| |_| 
       | |                                                                                          
       |_|                                                                                                  
            """
        )
        print(
            "\n Note: By default, the latest models are limited to 4,096 output tokens independent of the context window size!\n"
        )
        print("0. Terminate")
        print("1. Configure Paths")
        print(f"2. Set max_tokens (current: {max_tokens})")
        print("3. Create Batch Input File (.jsonl)")
        print("4. Use Existing Batch Input File (.jsonl)")
        print("5. Upload Batch Input File")
        print("6. Create and Submit Batch Request")
        print("7. Monitor Batch Processing")
        print("8. Download Batch Results")
        print("9. Save Responses as Markdown")
        print(f"10. Change API URL (current: {url})")
        print(f"11. Change Model (current: {model})")
        print("12. View Instruction Manual")
        print("13. Cancel Batch")
        print("14. List All Batches")
        print("15. Check OpenAI API Balance")

        choice = input("\nEnter your choice: \t ")

        try:
            if choice == "1":
                # Configure paths
                input_folder = (
                    input(f"Enter input folder (current: {input_folder}): ").strip()
                    or input_folder
                )
                batch_input_file = (
                    input(
                        f"Enter batch input file (current: {batch_input_file}): "
                    ).strip()
                    or batch_input_file
                )
                batch_output_file = (
                    input(
                        f"Enter batch output file (current: {batch_output_file}): "
                    ).strip()
                    or batch_output_file
                )
                output_folder = (
                    input(f"Enter output folder (current: {output_folder}): ").strip()
                    or output_folder
                )
                print(
                    f"Paths updated:\nInput Folder: {input_folder}\nBatch Input File: {batch_input_file}\nBatch Output File: {batch_output_file}\nOutput Folder: {output_folder}"
                )
            elif choice == "2":
                # Update max_tokens
                max_tokens_input = input(
                    "Enter maximum tokens for responses (current: {}): ".format(
                        max_tokens
                    )
                ).strip()
                if max_tokens_input.isdigit():
                    max_tokens = int(max_tokens_input)
                    print(f"`max_tokens` updated to {max_tokens}")
                else:
                    print("Invalid input. Please enter a positive integer.")
            elif choice == "3":
                # Create batch input file
                create_batch_input_jsonl(
                    input_folder,
                    batch_input_file,
                    model,
                    system_prompt,
                    user_prompt,
                    max_tokens,
                    url,
                )
            elif choice == "4":
                # Use existing batch input file
                batch_input_file = input(
                    "Enter the path of the existing .jsonl file: "
                ).strip()
                if not os.path.exists(batch_input_file):
                    print(f"File {batch_input_file} does not exist.")
                    logging.warning(
                        f"Attempted to use non-existent file: {batch_input_file}"
                    )
                else:
                    print(f"Using existing .jsonl file: {batch_input_file}")
                    logging.info(f"Using existing .jsonl file: {batch_input_file}")
            elif choice == "5":
                # Upload batch input file
                file_id = upload_batch_file(batch_input_file)
                print(f"Remember this File ID for later: {file_id}")
            elif choice == "6":
                # Create batch request
                file_id = input("Enter File ID: ")
                completion_window = (
                    input("Enter completion window (default: 24h): ").strip() or "24h"
                )
                metadata_description = input(
                    "Enter metadata description (default: 'Default batch job'): "
                ).strip()
                metadata = {"description": metadata_description or "Default batch job"}
                batch_id = create_batch_request(
                    file_id, completion_window=completion_window, metadata=metadata
                )
                print(f"Remember this Batch ID for later: {batch_id}")
            elif choice == "7":
                # Monitor batch processing
                batch_id = input("Enter Batch ID: ")
                result_file_id = monitor_batch(batch_id)
                print(f"Remember this Result File ID for later: {result_file_id}")
            elif choice == "8":
                # Download batch results
                result_file_id = input("Enter Result File ID: ")
                download_batch_results(result_file_id, batch_output_file)
            elif choice == "9":
                # Save responses as Markdown
                save_responses_as_markdown(batch_output_file, output_folder)
            elif choice == "10":
                # Change API URL
                url = input(
                    "Enter the new API URL (leave empty to reset to default): "
                ).strip()
                if not url:
                    url = "/v1/chat/completions"
                print(f"API URL updated to: {url}")
            elif choice == "11":
                # Change model
                model = input(
                    "Enter the new model (leave empty to reset to default): "
                ).strip()
                if not model:
                    model = "gpt-4-vision"
                print(f"Model updated to: {model}")
            elif choice == "12":
                # View instruction manual
                display_instruction_manual()
            elif choice == "13":
                # Cancel a batch
                batch_id = input("Enter Batch ID to cancel: ")
                cancel_batch(batch_id)
            elif choice == "14":
                # List all batches
                limit = input(
                    "Enter the number of batches to list (default: 10): "
                ).strip()
                limit = int(limit) if limit.isdigit() else 10
                after = (
                    input("Enter the cursor for pagination (optional): ").strip()
                    or None
                )
                list_batches(limit=limit, after=after)
            elif choice == "15":
                # Check OpenAI API Balance
                get_openai_balance()
            elif choice == "0":
                print("\nExiting the program...\n\nBye.")
                break
            else:
                print("Invalid choice. Please try again.")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if choice != "0":
                # Pause before returning to the menu
                input("\nPress Enter to return to the menu...")


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
