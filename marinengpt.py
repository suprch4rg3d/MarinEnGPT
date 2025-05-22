import os
import subprocess
from pathlib import Path
from helpers import *

# Get absolute path of the current directory (where this script lives)
BASE_DIR = Path(__file__).resolve().parent


def print_menu():
    clear_screen()
    print(
        rf"""
                                                             (_)
 __  __            _       _____        ____ ____ _____    }}--|--{{
|  \/  | __ _ _ __(_)_ __ | ____|_ __  / ___|  _ \_   _|  _   |   _
| |\/| |/ _` | '__| | '_ \|  _| | '_ \| |  _| |_) || |   `\__/ \__/`
| |  | | (_| | |  | | | | | |___| | | | |_| |  __/ | |     `-. .-'
|_|  |_|\__,_|_|  |_|_| |_|_____|_| |_|\____|_|    |_|        |           

            """
    )
    print("[1] Image Extractor Utility")
    # print("[2] OCR - Tesseract (Local)")
    print("[2] OCR - GPT-4o via OpenAI Chat Completion")
    print("[3] Generate Embeddings - OpenAI Embeddings API")
    print("[4] Load Embeddings into ChromaDB")
    print("[5] Launch Chainlit RAG Application")
    print()
    print("[q] Quit")
    print("=" * 60)


def run_subprocess(command_list):
    try:
        subprocess.run(command_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Command failed: {e}")
        input("\nPress Enter to return to menu...")


def main():
    while True:
        print_menu()
        choice = input("Select an option: ").strip().lower()

        if choice == "1":
            clear_screen()
            run_subprocess(
                ["poetry", "run", "python", str(BASE_DIR / "image_extractor.py")]
            )
        elif choice == "452":
            clear_screen()
            run_subprocess(
                ["poetry", "run", "python", str(BASE_DIR / "ocr_tesseract.py")]
            )
        elif choice == "2":
            clear_screen()
            run_subprocess(["poetry", "run", "python", str(BASE_DIR / "ocr_openai.py")])
        elif choice == "3":
            clear_screen()
            run_subprocess(
                ["poetry", "run", "python", str(BASE_DIR / "embeddings_openai.py")]
            )
        elif choice == "4":
            clear_screen()
            run_subprocess(
                ["poetry", "run", "python", str(BASE_DIR / "embeddings_openai.py")]
            )
        elif choice == "5":
            clear_screen()
            run_subprocess(
                ["poetry", "run", "chainlit", "run", str(BASE_DIR / "app.py"), "-w"]
            )
        elif choice in ["q", "quit", "exit"]:
            print("Exiting MarineN-GPT CLI Hub.")
            break
        else:
            print("Invalid choice. Try again.")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()
