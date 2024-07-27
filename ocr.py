import os
import ocrmypdf


def ocr_file(pdf_path):
    try:
        # Use OCRmyPDF API To Add OCR Layers, Correct Rotation, Deskew, And Detect Languages
        ocrmypdf.ocr(
            pdf_path,
            pdf_path,
            redo_ocr=False,
            force_ocr=True,
            rotate_pages=True,
            deskew=True,
            language=["eng", "chi_sim"],
        )

        # Check If OCR Was Added Successfully
        if os.path.exists(pdf_path + ".ocr.pdf"):
            print(f"OCR added successfully to {pdf_path}")
    except Exception as e:
        print(f"Error adding OCR to {pdf_path}: {str(e)}")


def ocr_generator(data_folder):
    # Create The 'data' Folder If It Doesn't Exist
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # If The Provided Path Is A Single File, OCR It Directly
    if os.path.isfile(data_folder):
        ocr_file(data_folder)
        return

    # Iterate Over The PDF Files In The 'data' Folder
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_folder, filename)
            print("###########################")
            print("" f"Adding OCR to {filename}")

            try:
                # Use OCRmyPDF API To Add OCR Layers, Correct Rotation, Deskew, And Detect Languages
                ocrmypdf.ocr(
                    pdf_path,
                    pdf_path,
                    redo_ocr=False,
                    force_ocr=True,
                    rotate_pages=True,
                    deskew=True,
                    language=["eng", "chi_sim"],
                )

                # Check If OCR Was Added Successfully
                if os.path.exists(pdf_path + ".ocr.pdf"):
                    print(f"OCR added successfully to {filename}")
            except Exception as e:
                print(f"Error adding OCR to {filename}: {str(e)}")


# Call The Function With The Desired Data Folder Path
data_folder = r"./data"
if __name__ == "__main__":
    ocr_generator(data_folder)
