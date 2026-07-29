import argparse
import os

import pandas as pd
import pyembroidery
from tqdm import tqdm


PROJECT_DIR = r"C:\Users\Asus\embroidery-finder"
REGISTRY_PATH = os.path.join(PROJECT_DIR, "designer_folders.csv")
OUTPUT_FOLDER = os.path.join(PROJECT_DIR, "dst_images")
CSV_PATH = os.path.join(PROJECT_DIR, "design_database.csv")


def load_database():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        if "designer" not in df.columns:
            df["designer"] = "Krishna"
        print(f"Existing database has {len(df)} entries")
        return df

    print("No existing database found, starting fresh")
    return pd.DataFrame(columns=["dst_path", "image_path", "design_name", "file_name", "designer"])


def load_registry(registry_path):
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Designer registry not found: {registry_path}")

    registry = pd.read_csv(registry_path)
    required_columns = {"designer", "folder"}
    missing_columns = required_columns - set(registry.columns)
    if missing_columns:
        raise ValueError(f"Designer registry is missing columns: {', '.join(sorted(missing_columns))}")

    registry = registry.dropna(subset=["designer", "folder"])
    registry["designer"] = registry["designer"].astype(str).str.strip()
    registry["folder"] = registry["folder"].astype(str).str.strip()
    registry = registry[(registry["designer"] != "") & (registry["folder"] != "")]
    return registry


def find_dst_files(folder):
    dst_files = []
    for root, _, files in os.walk(folder):
        for file_name in files:
            if file_name.lower().endswith(".dst"):
                dst_files.append(os.path.join(root, file_name))
    return sorted(dst_files)


def preview_path_for(designer, designer_folder, dst_path):
    relative_path = os.path.relpath(dst_path, designer_folder)
    safe_name = f"{designer}_" + relative_path.replace(os.sep, "_")
    safe_name = os.path.splitext(safe_name)[0] + ".png"
    return os.path.join(OUTPUT_FOLDER, safe_name)


def append_row(database, row):
    database = pd.concat([database, pd.DataFrame([row])], ignore_index=True)
    database.to_csv(CSV_PATH, index=False)
    return database


def scan_designer(database, existing_dst_paths, designer, folder):
    print(f"\nScanning: {folder}")
    print(f"Designer: {designer}")

    if not os.path.isdir(folder):
        print("Folder missing. Skipping.")
        return database, {
            "designer": designer,
            "found": 0,
            "skipped": 0,
            "added": 0,
            "failed": 0,
            "missing_folder": True,
        }

    dst_files = find_dst_files(folder)
    new_dst_files = [path for path in dst_files if path not in existing_dst_paths]

    print(f"Found {len(dst_files)} DST files")
    print(f"New files to process: {len(new_dst_files)}")
    print(f"Skipping (already indexed): {len(dst_files) - len(new_dst_files)}")

    added = 0
    failed = []

    if not new_dst_files:
        return database, {
            "designer": designer,
            "found": len(dst_files),
            "skipped": len(dst_files),
            "added": 0,
            "failed": 0,
            "missing_folder": False,
        }

    for dst_path in tqdm(new_dst_files, desc=f"Converting {designer}"):
        try:
            image_path = preview_path_for(designer, folder, dst_path)

            if not os.path.exists(image_path):
                pattern = pyembroidery.read(dst_path)
                pyembroidery.write(pattern, image_path)

            row = {
                "dst_path": dst_path,
                "image_path": image_path,
                "design_name": os.path.basename(os.path.dirname(dst_path)),
                "file_name": os.path.basename(dst_path),
                "designer": designer,
            }
            database = append_row(database, row)
            existing_dst_paths.add(dst_path)
            added += 1
        except Exception as exc:
            failed.append({"dst_path": dst_path, "error": str(exc)})

    if failed:
        print("Failed files:")
        for item in failed:
            print(f"  {item['dst_path']}: {item['error']}")

    return database, {
        "designer": designer,
        "found": len(dst_files),
        "skipped": len(dst_files) - len(new_dst_files),
        "added": added,
        "failed": len(failed),
        "missing_folder": False,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Scan all registered designer folders and add missing DST files."
    )
    parser.add_argument(
        "--registry",
        default=REGISTRY_PATH,
        help="CSV file with designer and folder columns.",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    database = load_database()
    existing_dst_paths = set(database["dst_path"].astype(str)) if "dst_path" in database.columns else set()
    registry = load_registry(args.registry)

    summaries = []
    for _, row in registry.iterrows():
        database, summary = scan_designer(
            database=database,
            existing_dst_paths=existing_dst_paths,
            designer=row["designer"],
            folder=row["folder"],
        )
        summaries.append(summary)

    print("\nSummary:")
    for item in summaries:
        if item["missing_folder"]:
            print(f"{item['designer']}: folder missing")
            continue
        print(
            f"{item['designer']}: found {item['found']}, "
            f"already indexed {item['skipped']}, "
            f"newly added {item['added']}, failed {item['failed']}"
        )

    print(f"\nTotal newly added: {sum(item['added'] for item in summaries)}")
    print(f"Total failed: {sum(item['failed'] for item in summaries)}")
    print(f"Total in database now: {len(database)}")


if __name__ == "__main__":
    main()
