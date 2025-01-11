import openai
from pydantic import List


def analyze_with_gpt4(text: str, images: List[str]) -> dict:
    """
    Use GPT-4's multimodal capabilities to analyze text and images together.
    """
    # Open the images
    image_files = [open(image_path, "rb") for image_path in images]

    # Create the prompt and the API call
    response = openai.ChatCompletion.create(
        model="gpt-4-vision",  
        messages=[
            {"role": "system", "content": "You are an expert in marine engineering."},
            {"role": "user", "content": f"Here is some text: {text}"},
        ],
        files=image_files,  # Pass the images for multimodal analysis
        max_tokens=3000,
        temperature=0
    )

    # Close the image files
    for img in image_files:
        img.close()

    # Extract structured data from the response
    structured_data = response['choices'][0]['message']['content']
    return structured_data