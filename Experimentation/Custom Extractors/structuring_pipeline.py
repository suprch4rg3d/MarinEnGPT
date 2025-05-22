import openai
from pydantic import ValidationError
import json
from ocr_tesseract import *
from image_extractor import *
from pydantic_dataclasses import *
from openai_clip import analyze_image_with_clip
from helpers import *

openai.api_key = os.environ.get("OPENAI_API_KEY")

# Text Structuring Function
def structure_text_with_openai(raw_text: str, image_descriptions: List[str]) -> dict:
    prompt = f"""
    You are an expert in marine engineering. I have extracted text and images from a marine engineering manual. 
    Please extract and structure the following information:

    1. Title of the manual
    2. Author(s)
    3. Publication date
    4. Sections with titles, start page, end page, a brief summary, and the content between start and end pages
    5. Language
    6. Ship types
    7. Keywords
    8. Safety information
    9. Specifications
    10. Image descriptions

    Here is the text:

    {raw_text}

    And here are the descriptions of diagrams/images:

    {json.dumps(image_descriptions, indent=2)}

    Please provide the extracted information in a structured JSON format.
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000,
        temperature=0
    )

    dd(response)
    structured_data = response['choices'][0]['message']['content'].strip()
    return json.loads(structured_data)
# Pipeline Execution
def main(pdf_path: str, output_txt_path: str, output_folder: str):
    # Perform OCR on the PDF (if needed)
    ocr_and_extract_text(pdf_path, output_txt_path)

    # Extract text from OCR'd PDF
    with open(output_txt_path, "r") as file:
        raw_text = file.read()

    # Extract images from PDF
    extract_images_from_pdf(pdf_path)
    image_paths = [str(p) for p in Path(output_folder).glob("*.png")]

    # Analyze images with CLIP
    image_descriptions = []
    for image_path in image_paths:
        description = analyze_image_with_clip(image_path)
        image_descriptions.append(description)

    # Structure text and image descriptions with OpenAI
    structured_data = structure_text_with_openai(raw_text, image_descriptions)

    # Validate and print structured data using Pydantic
    try:
        manual = MarineEngineeringManual(**structured_data)
        print(manual.json(indent=2))
    except (json.JSONDecodeError, ValidationError) as e:
        print("Error in structuring data:", e)

# Execute the pipeline
pdf_path = "./data/MF-050 Propeller fitting.pdf"
output_txt_path = "./manual_text.txt"
output_folder = "./extracted_images"
main(pdf_path, output_txt_path, output_folder)