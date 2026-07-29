import os
import argparse
import torch
import open_clip
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from config import DATABASE_CSV, QDRANT_PATH, DESIGN_COLLECTION as COLLECTION_NAME, VECTOR_SIZE
from view_type import classify_view_type

BATCH_SIZE = 64

parser = argparse.ArgumentParser(description="Fingerprint embroidery preview images for visual search.")
parser.add_argument(
    "--rebuild",
    action="store_true",
    help="Delete the visual search database and fingerprint every design again.",
)
args = parser.parse_args()

print("Loading AI model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
model.eval()
model = model.to(device)
print("AI model loaded!")

print("Setting up search database...")
os.makedirs(QDRANT_PATH, exist_ok=True)
client = QdrantClient(path=QDRANT_PATH)

def create_collection():
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    print(f"Search database created with {VECTOR_SIZE} dimensions!")

def collection_exists():
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
        return True
    except Exception:
        return False

if args.rebuild:
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
        print("Old collection deleted")
    except Exception:
        print("No old collection to delete")
    create_collection()
elif not collection_exists():
    create_collection()
else:
    print("Existing search database found. Only missing designs will be fingerprinted.")

def to_edges_3ch(pil_image, blur_kernel=9):
    img_array = np.array(pil_image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

df = pd.read_csv(DATABASE_CSV)
print(f"Database CSV rows: {len(df)}")

if "view_type" not in df.columns:
    print("No view_type column found - classifying now (see view_type.py)...")
    df["view_type"] = df["file_name"].apply(classify_view_type)
    df.to_csv(DATABASE_CSV, index=False)

def get_existing_point_ids():
    existing_ids = set()
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        existing_ids.update(int(record.id) for record in records)
        if offset is None:
            break
    return existing_ids

if args.rebuild:
    rows_to_fingerprint = df
else:
    existing_point_ids = get_existing_point_ids()
    print(f"Already fingerprinted: {len(existing_point_ids)}")
    rows_to_fingerprint = df.loc[~df.index.isin(existing_point_ids)]

print(f"New designs to fingerprint: {len(rows_to_fingerprint)}")

if rows_to_fingerprint.empty:
    print("\nDone!")
    print("Fingerprints created: 0")
    print("Failed: 0")
    print("AI search database is already up to date.")
    client.close()
    raise SystemExit(0)

points = []
failed = []
created = 0

for idx, row in tqdm(rows_to_fingerprint.iterrows(), total=len(rows_to_fingerprint), desc="Creating fingerprints"):
    try:
        image = Image.open(row['image_path']).convert('RGB')
        edge_image = to_edges_3ch(image)
        image_tensor = preprocess(edge_image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            fingerprint = features.cpu().numpy()[0].tolist()

        points.append(PointStruct(
            id=idx,
            vector=fingerprint,
            payload={
                'dst_path': row['dst_path'],
                'image_path': row['image_path'],
                'design_name': row['design_name'],
                'file_name': row['file_name'],
                'designer': row.get('designer', 'Krishna'),
                'view_type': row.get('view_type', 'other')
            }
        ))
        if len(points) >= BATCH_SIZE:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            created += len(points)
            points = []

    except Exception as e:
        failed.append({'path': row['image_path'], 'error': str(e)})

if points:
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    created += len(points)

print(f"\nDone!")
print(f"Fingerprints created: {created}")
print(f"Failed: {len(failed)}")
print("AI search database is ready! (Edge-detection mode)")
client.close()
