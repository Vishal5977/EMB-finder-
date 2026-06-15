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

def find_by_code(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")

    top_crop = img[0:int(h*0.25), 0:w]
    gray = cv2.cvtColor(top_crop, cv2.COLOR_BGR2GRAY)
    scaled = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    methods = {
        "scaled_gray": scaled,
        "scaled_inverted": cv2.bitwise_not(scaled),
        "scaled_threshold": cv2.threshold(scaled, 127, 255, cv2.THRESH_BINARY)[1],
        "scaled_threshold_inv": cv2.threshold(scaled, 127, 255, cv2.THRESH_BINARY_INV)[1],
    }

    all_codes = set()

    configs = ['--psm 11', '--psm 6', '--psm 3']

    for method_name, processed_img in methods.items():
        for config in configs:
            pil_img = Image.fromarray(processed_img)
            text = pytesseract.image_to_string(pil_img, config=config)
            codes = re.findall(r'\b\d{3,5}\b', text)

            if codes:
                print(f"--- {method_name} | {config} ---")
                print("Text:", text.strip()[:150])
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