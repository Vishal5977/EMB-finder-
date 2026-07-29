import cv2
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

DATABASE_CSV = r'C:\Users\Asus\embroidery-finder\design_database.csv'
CUSTOMER_IMAGE = r'C:\Users\Asus\embroidery-finder\test_customer.jpg'
BATCH_SIZE = 24


def customer_edges(image):
    array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 80, 200)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))


def database_edges(image):
    array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))


def letterbox(image):
    width, height = image.size
    side = max(width, height)
    canvas = Image.new('RGB', (side, side), 'black')
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
model.eval().to(device)

def encode_batch(images):
    tensors = torch.stack([preprocess(image) for image in images]).to(device)
    with torch.no_grad():
        features = model.encode_image(tensors)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu()

query = encode_batch([letterbox(customer_edges(Image.open(CUSTOMER_IMAGE)))])[0]
df = pd.read_csv(DATABASE_CSV)
scores = []

for start in range(0, len(df), BATCH_SIZE):
    batch = df.iloc[start:start + BATCH_SIZE]
    images = []
    valid_rows = []
    for index, row in batch.iterrows():
        try:
            images.append(letterbox(database_edges(Image.open(row['image_path']))))
            valid_rows.append((index, row))
        except Exception as error:
            print('FAILED:', row['image_path'], error)
    if images:
        features = encode_batch(images)
        similarities = features @ query
        for (_, row), score in zip(valid_rows, similarities.tolist()):
            scores.append((score, row['designer'], row['design_name'], row['file_name']))
    print(f'Processed {min(start + BATCH_SIZE, len(df))}/{len(df)}')

scores.sort(reverse=True, key=lambda item: item[0])
print('\nTOP 30')
for rank, (score, designer, design_name, file_name) in enumerate(scores[:30], 1):
    print(rank, round(score * 100, 2), designer, design_name, file_name)

print('\nALL 2660 RESULTS')
for rank, item in enumerate(scores, 1):
    score, designer, design_name, file_name = item
    if '2660' in str(design_name):
        print(rank, round(score * 100, 2), designer, design_name, file_name)
