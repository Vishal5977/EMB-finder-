import pytesseract
from PIL import Image
import pandas as pd
import re
import sys
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

DATABASE_CSV = r'C:\Users\Asus\embroidery-finder\design_database.csv'
df = pd.read_csv(DATABASE_CSV)
valid_codes = set(df['design_name'].astype(str).unique())

def find_by_code(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")
    print(f"Valid codes in DB: {sorted(valid_codes)}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    variants = {
        "gray": scaled,
        "inverted": cv2.bitwise_not(scaled),
        "otsu": cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        "otsu_inv": cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        "adaptive": cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
        "adaptive_inv": cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2),
    }

    configs = ['--psm 11', '--psm 6', '--psm 12']

    all_codes_found = {}

    for variant_name, processed_img in variants.items():
        for config in configs:
            pil_img = Image.fromarray(processed_img)
            text = pytesseract.image_to_string(pil_img, config=config)
            codes = re.findall(r'\b\d{3,5}\b', text)

            for code in codes:
                all_codes_found[code] = all_codes_found.get(code, 0) + 1

    print("=" * 50)
    print("ALL CODES FOUND (with frequency):", all_codes_found)

    valid_matches = [code for code in all_codes_found if code in valid_codes]

    print("=" * 50)
    if valid_matches:
        valid_matches.sort(key=lambda c: all_codes_found[c], reverse=True)
        best_match = valid_matches[0]
        print(f"BEST MATCH: {best_match} (detected {all_codes_found[best_match]} times)")
        print(f"All valid candidates: {valid_matches}")
        print()
        result_rows = df[df['design_name'].astype(str) == best_match]
        for idx, row in result_rows.iterrows():
            print(f"  - {row['file_name']} -> {row['dst_path']}")
    else:
        print("NO EXACT CODE MATCH FOUND IN DATABASE")
        print("Raw codes detected (none matched DB):", list(all_codes_found.keys()))

if __name__ == "__main__":
    image_path = sys.argv[1]
    find_by_code(image_path)