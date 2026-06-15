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

number_to_designs = {}
for design_name in df['design_name'].astype(str).unique():
    numbers_in_name = re.findall(r'\d{3,5}', design_name)
    for num in numbers_in_name:
        if num not in number_to_designs:
            number_to_designs[num] = set()
        number_to_designs[num].add(design_name)

COLOR_RANGES = {
    "pink_magenta": ([130, 40, 40], [175, 255, 255]),
    "white": ([0, 0, 120], [180, 80, 255]),
    "yellow": ([20, 50, 50], [35, 255, 255]),
    "cyan": ([85, 50, 50], [105, 255, 255]),
    "red": ([0, 50, 50], [10, 255, 255]),
    "red2": ([170, 50, 50], [180, 255, 255]),
    "green": ([40, 50, 50], [80, 255, 255]),
    "orange": ([10, 50, 50], [20, 255, 255]),
    "purple_blue": ([105, 50, 50], [130, 255, 255]),
}

def find_code_in_image(image_path_or_cv2):
    if isinstance(image_path_or_cv2, str):
        img = cv2.imread(image_path_or_cv2)
        if img is None:
            return None, None
    else:
        img = image_path_or_cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    all_codes_found = {}

    for color_name, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        if cv2.countNonZero(mask) < 100:
            continue
        for config in ['--psm 11', '--psm 6']:
            text = pytesseract.image_to_string(mask, config=config)
            codes = re.findall(r'\b\d{3,5}\b', text)
            for code in codes:
                all_codes_found[code] = all_codes_found.get(code, 0) + 1

    valid_matches = []
    for ocr_num, freq in all_codes_found.items():
        if ocr_num in number_to_designs:
            for design_name in number_to_designs[ocr_num]:
                valid_matches.append((ocr_num, design_name, freq))

    if valid_matches:
        valid_matches.sort(key=lambda x: x[2], reverse=True)
        seen_designs = []
        for ocr_num, design_name, freq in valid_matches:
            if design_name not in seen_designs:
                seen_designs.append(design_name)

        best_design = seen_designs[0]
        result_rows = df[df['design_name'].astype(str) == best_design]
        return best_design, result_rows
    else:
        return None, None


if __name__ == "__main__":
    image_path = sys.argv[1]
    code, rows = find_code_in_image(image_path)
    if code:
        print(f"MATCH FOUND: {code}")
        for idx, row in rows.iterrows():
            print(f"  - {row['file_name']} -> {row['dst_path']}")
    else:
        print("NO MATCH FOUND")