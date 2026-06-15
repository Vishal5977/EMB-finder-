import os
import torch
import open_clip
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

DATABASE_CSV = r'C:\Users\Asus\embroidery-finder\design_database.csv'
QDRANT_PATH = r'C:\Users\Asus\embroidery-finder\qdrant_db'

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

try:
    client.delete_collection(collection_name="embroidery_designs")
    print("Old collection deleted")
except Exception as e:
    print("No old collection to delete")

client.create_collection(
    collection_name="embroidery_designs",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)
print("Search database created with 768 dimensions!")

def to_edges_3ch(pil_image):
    img_array = np.array(pil_image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

df = pd.read_csv(DATABASE_CSV)
print(f"Found {len(df)} designs to fingerprint")

points = []
failed = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating fingerprints"):
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
                'designer': row.get('designer', 'Krishna')
            }
        ))

    except Exception as e:
        failed.append({'path': row['image_path'], 'error': str(e)})

print("Saving fingerprints to search database...")
client.upsert(
    collection_name="embroidery_designs",
    points=points
)

print(f"\nDone!")
print(f"Fingerprints created: {len(points)}")
print(f"Failed: {len(failed)}")
print("AI search database is ready! (Edge-detection mode)")