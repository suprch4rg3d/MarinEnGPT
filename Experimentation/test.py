# def main(pdf_path: str, output_txt_path: str, output_folder: str):
#     # Step 1: Perform OCR on the PDF (if needed)
#     ocr_file(pdf_path)

#     # Step 2: Extract text from OCR'd PDF
#     with open(output_txt_path, "r") as file:
#         raw_text = file.read()

#     # Step 3: Extract images from PDF
#     extract_images_from_pdf(pdf_path)
#     image_paths = [str(p) for p in Path(output_folder).glob("*.png")]

#     # Step 4: Analyze text and images with GPT-4 Multimodal
#     structured_data = analyze_with_gpt4(raw_text, image_paths)

#     # Step 5: Validate and print structured data using Pydantic
#     try:
#         manual = MarineEngineeringManual(**structured_data)
#         print(manual.json(indent=2))
#     except (json.JSONDecodeError, ValidationError) as e:
#         print("Error in structuring data:", e)

# # Execute the pipeline
# pdf_path = "manual.pdf"
# output_txt_path = "manual_text.txt"
# output_folder = "./extracted_images"
# main(pdf_path, output_txt_path, output_folder)