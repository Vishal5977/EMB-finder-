import pytesseract
from PIL import Image
import pandas as pd
import re
import sys

# Tell pytesseract where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load our design database
DATABASE_CSV = r'C:\Users\Asus\embroidery-finder\design_database.csv'
df = pd.read_csv(DATABASE_CSV)

def find_by_code(image_path):
    # Read text from image
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)

    print("=" * 50)
    print("TEXT FOUND IN IMAGE:")
    print(text)
    print("=" * 50)

    # Find numbers that look like design codes (3-5 digits)
    codes_found = re.findall(r'\b\d{3,5}\b', text)
    print("POSSIBLE CODES FOUND:", codes_found)

    # Search database for matching design_name
    matches = []
    for code in codes_found:
        matching_rows = df[df['design_name'].astype(str) == code]
        if len(matching_rows) > 0:
            matches.append(code)

    print("=" * 50)
    if matches:
        print(f"MATCH FOUND! Design code: {matches[0]}")
        print("\nDST Files for this design:")
        result_rows = df[df['design_name'].astype(str) == matches[0]]
        for idx, row in result_rows.iterrows():
            print(f"  - {row['file_name']} -> {row['dst_path']}")
    else:
        print("NO EXACT CODE MATCH FOUND IN DATABASE")

if __name__ == "__main__":
    image_path = sys.argv[1]
    find_by_code(image_path)