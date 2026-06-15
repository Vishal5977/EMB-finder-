import os
import pyembroidery
import pandas as pd
from tqdm import tqdm
#py -3.11 "C:\Users\Asus\embroidery-finder\scan_dst.py"
DESIGNS_FOLDER = r'C:\Users\Asus\all pendrive design\dual hybrid\Varsha Creations'
DESIGNER_NAME = 'Varsha Creations'

OUTPUT_FOLDER = r'C:\Users\Asus\embroidery-finder\dst_images'
CSV_PATH = r'C:\Users\Asus\embroidery-finder\design_database.csv'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if os.path.exists(CSV_PATH):
    existing_df = pd.read_csv(CSV_PATH)
    if 'designer' not in existing_df.columns:
        existing_df['designer'] = 'Krishna'
    existing_dst_paths = set(existing_df['dst_path'].astype(str))
    print(f"Existing database has {len(existing_df)} entries")
else:
    existing_df = pd.DataFrame(columns=['dst_path', 'image_path', 'design_name', 'file_name', 'designer'])
    existing_dst_paths = set()
    print("No existing database found, starting fresh")

print(f"\nScanning: {DESIGNS_FOLDER}")
print(f"Designer: {DESIGNER_NAME}")

dst_files = []
for root, dirs, files in os.walk(DESIGNS_FOLDER):
    for file in files:
        if file.lower().endswith('.dst'):
            dst_files.append(os.path.join(root, file))

print(f"Found {len(dst_files)} DST files in this folder")

new_dst_files = [f for f in dst_files if f not in existing_dst_paths]
print(f"New files to process: {len(new_dst_files)}")
print(f"Skipping (already indexed): {len(dst_files) - len(new_dst_files)}")

if len(new_dst_files) == 0:
    print("\nNothing new to add. Done!")
else:
    results = []
    failed = []

    for dst_path in tqdm(new_dst_files, desc="Converting DST files"):
        try:
            relative_path = os.path.relpath(dst_path, DESIGNS_FOLDER)
            safe_name = f"{DESIGNER_NAME}_" + relative_path.replace(os.sep, '_').replace('.DST', '.png').replace('.dst', '.png')
            image_path = os.path.join(OUTPUT_FOLDER, safe_name)

            pattern = pyembroidery.read(dst_path)
            pyembroidery.write(pattern, image_path)

            results.append({
                'dst_path': dst_path,
                'image_path': image_path,
                'design_name': os.path.basename(os.path.dirname(dst_path)),
                'file_name': os.path.basename(dst_path),
                'designer': DESIGNER_NAME
            })
        except Exception as e:
            failed.append({'dst_path': dst_path, 'error': str(e)})

    new_df = pd.DataFrame(results)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_csv(CSV_PATH, index=False)

    print(f"\nDone!")
    print(f"Newly added: {len(results)}")
    print(f"Failed: {len(failed)}")
    print(f"Total in database now: {len(combined_df)}")
    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  {f['dst_path']}: {f['error']}")