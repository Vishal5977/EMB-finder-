import pytesseract
from PIL import Image, ImageOps
import pandas as pd
import re
import sys
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

DATABASE_CSV = r'C:\Users\Asus\embroidery-finder\design_database.csv'
df = pd.read_csv(DATABASE_CSV)

def find_by_code(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    methods = {
        "original_gray": gray,
        "inverted": cv2.bitwise_not(gray),
        "threshold": cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1],
        "threshold_inv": cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)[1],
    }

    all_codes = set()

    for method_name, processed_img in methods.items():
        pil_img = Image.fromarray(processed_img)
        text = pytesseract.image_to_string(pil_img)
        codes = re.findall(r'\b\d{3,5}\b', text)

        print(f"--- Method: {method_name} ---")
        print("Text:", text.strip()[:200])
        print("Codes found:", codes)
        print()

        all_codes.update(codes)

    print("=" * 50)
    print("ALL POSSIBLE CODES:", all_codes)

    matches = []
    for code in all_codes:
        matching_rows = df[df['design_name'].astype(str) == code]
        if len(matching_rows) > 0:
            matches.append(code)

    print("=" * 50)
    if matches:
        for match_code in matches:
            print(f"MATCH FOUND! Design code: {match_code}")
            result_rows = df[df['design_name'].astype(str) == match_code]
            for idx, row in result_rows.iterrows():
                print(f"  - {row['file_name']} -> {row['dst_path']}")
    else:
        print("NO EXACT CODE MATCH FOUND IN DATABASE")

if __name__ == "__main__":
    image_path = sys.argv[1]
    find_by_code(image_path)