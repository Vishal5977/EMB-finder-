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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    all_codes_found = {}

    color_ranges = {
        "pink_magenta": ([130, 40, 40], [175, 255, 255]),
        "white": ([0, 0, 180], [180, 60, 255]),
        "yellow": ([20, 50, 50], [35, 255, 255]),
        "cyan": ([85, 50, 50], [105, 255, 255]),
    }

    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        if cv2.countNonZero(mask) < 100:
            continue

        for config in ['--psm 11', '--psm 6']:
            text = pytesseract.image_to_string(mask, config=config)
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