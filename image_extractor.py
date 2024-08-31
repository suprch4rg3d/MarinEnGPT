import os
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def embed_metadata(image_path, filename, page_label):
    image = Image.open(image_path)
    metadata = PngInfo()
    metadata.add_text('file_name', filename)
    metadata.add_text('page_label', str(page_label))
    metadata.add_text('image_path', str(image_path))
    
    image.save(image_path, pnginfo=metadata)
    image.close()

def extract_images_from_pdf(pdf_file):
    # Extract PDF Filename Without Extension
    filename = Path(pdf_file).stem
        
    # Create The 'images' Folder If It Does Not Exist
    # images_folder = Path(output_folder) / "images"
    # images_folder.mkdir(parents=True, exist_ok=True)
    
    # Get The Parent Directory Of The PDF File
    output_folder = Path(pdf_file).parent
     
    # Open The PDF File
    pdf_document = fitz.open(pdf_file)

    # Iterate Through Each Page
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)

        # Extract Images From The Page
        image_list = page.get_images()
        for image_index, image in enumerate(image_list):
            xref = image[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]

            # Save The Image To A File In The Images Folder
            image_path = output_folder / f"{filename}_page_{page_number + 1}_image_{image_index + 1}.png"
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            # Embed Metadata Into The Image (This Is Not Used From RAG)
            embed_metadata(image_path, filename, page_number + 1)

            print(f"Image saved: {image_path}")

    # Close The PDF Document
    pdf_document.close()

def extract_images_from_pdfs_in_folder(folder_path):
    # Get The Parent Directory Of The folder_path
    # output_folder = os.path.dirname(folder_path)
    
    # Iterate Through Each File In The Folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_file = Path(folder_path) / filename
            # output_folder = folder_path
            extract_images_from_pdf(pdf_file)

# Example Usage
# folder_path = r"./Data"
# extract_images_from_pdfs_in_folder(folder_path)

#####################################################################

def check_embedded_metadata(image_path):
    check_image = Image.open(image_path)
    metadata = check_image.text
    check_image.close()
    return metadata

# Example Usage
# metadata = check_embedded_metadata("./Data/Images/MF-194 Instruction manual for Fe／Cu-Ions generating system_page_2_image_1.png")
# print(metadata)