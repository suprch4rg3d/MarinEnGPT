import os
import base64
import json
import openai
import time
import logging
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


def display_instruction_manual():
    """
    Displays a detailed instruction manual for the user.
    """
    manual = """
=== Instruction Manual ===

1. Configure Paths
   - Update the paths for input folder, batch input file, output file, and output folder.

2. Create Batch Input File (.jsonl)
   - Generate a `.jsonl` file from images in the specified input folder.

3. Use Existing Batch Input File (.jsonl)
   - Reuse a pre-existing `.jsonl` file without recreating it.

4. Upload Batch Input File
   - Upload the `.jsonl` file to OpenAI for batch processing.

5. Create and Submit Batch Request
   - Use the File ID to submit a batch processing job.

6. Monitor Batch Processing
   - Check the status of your batch job using the Batch ID.

7. Download Batch Results
   - Download processed results using the Result File ID.

8. Save Responses as Markdown
   - Save responses from the results file into Markdown format.

9. Change API URL
   - Dynamically update the API endpoint URL.

10. Change Model
   - Update the model being used for processing.

11. View Instruction Manual
   - Display this detailed guide.

12. Exit
   - Exit the program.

===========================
"""
    print(manual)
    logging.info("Displayed the instruction manual.")


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

        choice = input("\nEnter your choice: ")

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
                input("\nPress Enter or Space to return to the menu...")
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
            elif choice == "0":
                # Exit
                print("Exiting. Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")
        except Exception as e:
            print(f"An error occurred: {e}")


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
