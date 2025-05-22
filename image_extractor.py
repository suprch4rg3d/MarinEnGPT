import os
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
from tqdm import tqdm  # For Progress Tracking
import logging

# Configuration for Image Preprocessing
default_resize_factor = 1.5
default_grayscale = True
preprocessed_prefix = "P_"  # Prefix For Preprocessed Images and Folders
logging_enabled = False  # Global Flag for Logging

# Setup Logging
logging.basicConfig(filename="image_extractor.log", level=logging.INFO, format="%(asctime)s - %(message)s")

def log(message):
    """
    Log a message if logging is enabled.

    Args:
        message (str): The message to log.
    """
    if logging_enabled:
        logging.info(message)

def toggle_logging():
    """
    Enable or disable logging from the menu.
    """
    global logging_enabled
    choice = input(f"Logging is currently {'enabled' if logging_enabled else 'disabled'}. Toggle? (yes/no): ").strip().lower()
    if choice == 'yes':
        logging_enabled = not logging_enabled
        print(f"Logging {'enabled' if logging_enabled else 'disabled'}.")

def configure_preprocessing():
    """
    Configure preprocessing options such as resize factor and grayscale conversion.
    """
    global default_resize_factor, default_grayscale
    print("\nConfigure Preprocessing Options:")
    try:
        resize_factor = input(f"Enter resize factor (current: {default_resize_factor}): ").strip()
        if resize_factor:
            default_resize_factor = float(resize_factor)
        
        grayscale = input(f"Enable grayscale conversion? (yes/no, current: {'yes' if default_grayscale else 'no'}): ").strip().lower()
        if grayscale in ['yes', 'no']:
            default_grayscale = (grayscale == 'yes')

        print(f"\nUpdated Preprocessing Settings:\nResize Factor: {default_resize_factor}\nGrayscale: {'Enabled' if default_grayscale else 'Disabled'}")
        log(f"Updated preprocessing settings: Resize Factor={default_resize_factor}, Grayscale={'Enabled' if default_grayscale else 'Disabled'}")
    except Exception as e:
        print(f"Error configuring preprocessing options: {e}")
        log(f"Error configuring preprocessing options: {e}")

def show_instructions():
    """
    Display a short instruction manual.
    """
    instructions = """
    Instruction Manual:
    -------------------
    1. Configure Preprocessing Options:
       - Adjust resize factor and enable/disable grayscale for image preprocessing.
        
        ** Resize Factor indicates how much an image is scaled relative to its original dimensions.
            >A resize factor greater than 1 means enlarging the image (e.g., a factor of 2 doubles the size).
            >A resize factor less than 1 means reducing the image size (e.g., a factor of 0.5 halves the size).

    2. Enable/Disable Logging:
       - Enable or disable logging of all actions to 'image_extractor.log'.

    3. Extract Images from a Single PDF:
       - Extract embedded images from a PDF file.
       - Optionally preprocess images (resize and grayscale).

    4. Extract Pages of a PDF as Images:
       - Save each page of a PDF as an image.
       - Optionally preprocess images.

    5. Extract Images from All PDFs in a Folder:
       - Process all PDF files in a specified folder.
       - Extract embedded images or save pages as images.
       - Optionally preprocess images.

    6. Check Metadata:
       - View metadata embedded in PNG images.

    7. Logging:
       - Logs are stored in 'image_extractor.log' for debugging or review.

    8. Exiting:
       - Select the exit option to close the program.

    """
    print(instructions)

def embed_metadata(image_path, filename, page_label):
    """
    Embed metadata into a PNG image.

    Args:
        image_path (Path): Path to the image file.
        filename (str): Original PDF filename without extension.
        page_label (int): Page number of the PDF.
    """
    try:
        with Image.open(image_path) as image:
            metadata = PngInfo()
            metadata.add_text('file_name', filename)
            metadata.add_text('page_label', str(page_label))
            metadata.add_text('image_path', str(image_path))
            
            image.save(image_path, pnginfo=metadata)
        print(f"Metadata embedded: {image_path}")
        log(f"Metadata embedded into {image_path}")
    except Exception as e:
        print(f"Error embedding metadata into {image_path}: {e}")
        log(f"Error embedding metadata into {image_path}: {e}")

def preprocess_image(image_path):
    """
    Preprocess an image by resizing, optionally converting to grayscale.

    Args:
        image_path (Path): Path to the image file.
    """
    try:
        with Image.open(image_path) as image:
            image = image.resize((int(image.width * default_resize_factor), int(image.height * default_resize_factor)))  # Resize
            if default_grayscale:
                image = ImageOps.grayscale(image)  # Convert to Grayscale If Enabled
            image.save(image_path)  # Overwrite the Original Image
        print(f"Image preprocessed: {image_path}")
        log(f"Image preprocessed: {image_path}")
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        log(f"Error preprocessing image {image_path}: {e}")

def extract_pages_as_images(pdf_file, output_folder, preprocess=False):
    """
    Extract pages of a PDF file as images, optionally preprocess them, and embed metadata.

    Args:
        pdf_file (Path): Path to the PDF file.
        output_folder (Path): Folder to save extracted images.
        preprocess (bool): Whether to preprocess the extracted images.
    """
    try:
        # Extract PDF Filename Without Extension
        filename = pdf_file.stem

        # Modify Output Folder Name for Preprocessed Files
        if preprocess:
            output_folder = output_folder / f"{preprocessed_prefix}{filename}"
        else:
            output_folder = output_folder / filename

        output_folder.mkdir(parents=True, exist_ok=True)

        # Open The PDF File
        pdf_document = fitz.open(pdf_file)
        num_pages = len(pdf_document)

        # Iterate Through Each Page
        for page_number in tqdm(range(num_pages), desc="Processing Pages", unit="page"):
            page = pdf_document.load_page(page_number)
            pix = page.get_pixmap()

            # Save Each Page as an Image
            image_name = f"{preprocessed_prefix if preprocess else ''}{filename}_page_{page_number + 1}.png"
            image_path = output_folder / image_name
            pix.save(image_path)

            # Optionally Preprocess Image
            if preprocess:
                preprocess_image(image_path)

            # Embed Metadata Into The Image
            embed_metadata(image_path, filename, page_number + 1)

        # Close The PDF Document
        pdf_document.close()
        log(f"Pages extracted as images from {pdf_file}, preprocess={preprocess}")
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")
        log(f"Error processing {pdf_file}: {e}")

def extract_images_from_pdf(pdf_file, output_folder, preprocess=False):
    """
    Extract images from a PDF file and save them to a specified folder, optionally preprocess them.

    Args:
        pdf_file (Path): Path to the PDF file.
        output_folder (Path): Folder to save extracted images.
        preprocess (bool): Whether to preprocess the extracted images.
    """
    try:
        # Extract PDF Filename Without Extension
        filename = pdf_file.stem

        # Modify output folder name for preprocessed files
        if preprocess:
            output_folder = output_folder / f"{preprocessed_prefix}{filename}"
        else:
            output_folder = output_folder / filename

        output_folder.mkdir(parents=True, exist_ok=True)

        # Open The PDF File
        pdf_document = fitz.open(pdf_file)

        # Iterate Through Each Page
        for page_number in tqdm(range(len(pdf_document)), desc="Processing Pages", unit="page"):
            page = pdf_document.load_page(page_number)

            # Extract Images From The Page
            image_list = page.get_images(full=True)
            for image_index, image in enumerate(image_list):
                xref = image[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]

                # Save The Image To A File In The Images Folder
                image_name = f"{preprocessed_prefix if preprocess else ''}{filename}_page_{page_number + 1}_image_{image_index + 1}.png"
                image_path = output_folder / image_name
                with open(image_path, "wb") as image_file:
                    image_file.write(image_bytes)

                # Optionally Preprocess Image
                if preprocess:
                    preprocess_image(image_path)

                # Embed Metadata Into The Image
                embed_metadata(image_path, filename, page_number + 1)

        # Close The PDF Document
        pdf_document.close()
        log(f"Images extracted from {pdf_file}, preprocess={preprocess}")
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")
        log(f"Error processing {pdf_file}: {e}")

def extract_images_from_pdfs_in_folder(folder_path, output_folder, preprocess=False):
    """
    Extract images from all PDFs in a specified folder, optionally preprocess them.

    Args:
        folder_path (Path): Folder containing PDF files.
        output_folder (Path): Folder to save all extracted images.
        preprocess (bool): Whether to preprocess the extracted images.
    """
    try:
        folder_path = Path(folder_path)
        output_folder = Path(output_folder)

        # Ensure Output Folder Exists
        output_folder.mkdir(parents=True, exist_ok=True)
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

        for filename in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
            pdf_file = folder_path / filename
            extract_images_from_pdf(pdf_file, output_folder, preprocess=preprocess)
        log(f"Processed all PDFs in {folder_path}, preprocess={preprocess}")
    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")
        log(f"Error processing folder {folder_path}: {e}")

def check_embedded_metadata(image_path):
    """
    Check and return metadata embedded in a PNG image.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        dict: Dictionary of embedded metadata.
    """
    try:
        with Image.open(image_path) as check_image:
            metadata = check_image.text
        log(f"Metadata checked for {image_path}")
        return metadata
    except Exception as e:
        print(f"Error reading metadata from {image_path}: {e}")
        log(f"Error reading metadata from {image_path}: {e}")
        return {}

def menu():
    while True:
        print(r'''
___  ___              _         _____        _____ ______  _____                         
|  \/  |             (_)       |  ___|      |  __ \| ___ \|_   _|                        
| .  . |  __ _  _ __  _  _ __  | |__  _ __  | |  \/| |_/ /  | |                          
| |\/| | / _` || '__|| || '_ \ |  __|| '_ \ | | __ |  __/   | |                          
| |  | || (_| || |   | || | | || |___| | | || |_\ \| |      | |                          
\______/ \__,_||_|   |_||_| |_|\____/|_____| \_____\_|      \_/          _               
|_   _|                              |  ___|     | |                    | |              
  | |  _ __ ___    __ _   __ _   ___ | |__ __  __| |_  _ __  __ _   ___ | |_  ___   _ __ 
  | | | '_ ` _ \  / _` | / _` | / _ \|  __|\ \/ /| __|| '__|/ _` | / __|| __|/ _ \ | '__|
 _| |_| | | | | || (_| || (_| ||  __/| |___ >  < | |_ | |  | (_| || (__ | |_| (_) || |   
 \___/|_| |_| |_| \__,_| \__, | \___|\____//_/\_\ \__||_|   \__,_| \___| \__|\___/ |_|   
                          __/ |                                                          
                         |___/                                                           
            ''')
        print("\n Note: For certain PDF files, which are not digitally born, some options may have the same effect!\n")
        print("0. Terminate")
        print("1. Configure preprocessing options")
        print("2. Enable/Disable logging")
        print("3. Show instruction manual")
        print("4. Extract images from a single PDF (without preprocessing)")
        print("5. Extract images from a single PDF (with preprocessing)")
        print("6. Extract pages of a PDF as images (without preprocessing)")
        print("7. Extract pages of a PDF as images (with preprocessing)")
        print("8. Extract images from all PDFs in a folder (without preprocessing)")
        print("9. Extract images from all PDFs in a folder (with preprocessing)")
        print("10. Check metadata of an image")

        choice = input("\nEnter your choice (0-10): \t")

        if choice == "1":
            configure_preprocessing()
        elif choice == "2":
            toggle_logging()
        elif choice == "3":
            show_instructions()
            input("\nPress Enter or Space to return to the menu...")
        elif choice == "4":
            pdf_file = input("Enter the path to the PDF file: ").strip()
            output_folder = input("Enter the output folder path: ").strip()
            extract_images_from_pdf(Path(pdf_file), Path(output_folder), preprocess=False)
            input("\nPress Enter or Space to return to the menu...")
        elif choice == "5":
            pdf_file = input("Enter the path to the PDF file: ").strip()
            output_folder = input("Enter the output folder path: ").strip()
            extract_images_from_pdf(Path(pdf_file), Path(output_folder), preprocess=True)
            input("\nPress Enter or Space to return to the menu...")
        elif choice == "6":
            pdf_file = input("Enter the path to the PDF file: ").strip()
            output_folder = input("Enter the output folder path: ").strip()
            extract_pages_as_images(Path(pdf_file), Path(output_folder), preprocess=False)
            input("\nPress Enter or Space to return to the menu...")
        elif choice == "7":
            pdf_file = input("Enter the path to the PDF file: ").strip()
            output_folder = input("Enter the output folder path: ").strip()
            extract_pages_as_images(Path(pdf_file), Path(output_folder), preprocess=True)
            input("\nPress Enter or Space to return to the menu...")
        elif choice == "8":
            folder_path = input("Enter the folder containing PDFs: ").strip()
            output_folder = input("Enter the output folder path: ").strip()
            extract_images_from_pdfs_in_folder(folder_path, output_folder, preprocess=False)
            input("\nPress Enter or Space to return to the menu...")
        elif choice == "9":
            folder_path = input("Enter the folder containing PDFs: ").strip()
            output_folder = input("Enter the output folder path: ").strip()
            extract_images_from_pdfs_in_folder(folder_path, output_folder, preprocess=True)
            input("\nPress Enter or Space to return to the menu...")
        elif choice == "10":
            image_path = input("Enter the path to the image file: ").strip()
            metadata = check_embedded_metadata(Path(image_path))
            print("\nMetadata:", metadata)
            input("\nPress Enter or Space to return to the menu...")
        elif choice == "0":
            print("\nExiting the program...\n\nBye.")
            break
        else:
            print("Invalid choice. Please try again.")

# Entry point
if __name__ == "__main__":
    menu()