import cv2
import numpy as np
import open_clip
import torch
from PIL import Image
from qdrant_client import QdrantClient

ROOT = r'C:\Users\Asus\embroidery-finder'
IMAGE_PATH = ROOT + r'\test_customer.jpg'
QDRANT_PATH = ROOT + r'\qdrant_db'

rgb = np.array(Image.open(IMAGE_PATH).convert('RGB'))
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
current = cv2.Canny(cv2.GaussianBlur(gray, (7, 7), 0), 80, 200)

r, g, b = cv2.split(rgb)
dark = ((r < 125) & (g < 85) & (b < 125)).astype(np.uint8) * 255
dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
count, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
blouse = np.zeros_like(dark)
for component in (np.argsort(stats[1:, cv2.CC_STAT_AREA])[-3:] + 1 if count > 1 else []):
    blouse[labels == component] = 255
near_blouse = cv2.dilate(blouse, np.ones((25, 25), np.uint8))
hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
mask = ((hsv[:, :, 2] > 125) & (near_blouse > 0)).astype(np.uint8) * 255
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

# Thin the isolated regions back to embroidery-like contours.
refined = cv2.Canny(cv2.GaussianBlur(mask, (5, 5), 0), 40, 120)

variants = {'CURRENT_EDGES': current, 'ISOLATED_MASK': mask, 'REFINED_MASK_EDGES': refined}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
model.eval().to(device)
client = QdrantClient(path=QDRANT_PATH)

for label, image_array in variants.items():
    image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB))
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    results = client.query_points(
        collection_name='embroidery_designs',
        query=features.cpu().numpy()[0].tolist(),
        limit=973,
    ).points
    print('\n' + label)
    for rank, result in enumerate(results[:10], 1):
        print(rank, round(result.score * 100, 2), result.payload.get('designer'), result.payload.get('design_name'), result.payload.get('file_name'))
    matches = [(rank, result) for rank, result in enumerate(results, 1) if '2660' in str(result.payload.get('design_name'))]
    print('2660:')
    for rank, result in matches:
        print(rank, round(result.score * 100, 2), result.payload.get('file_name'))

client.close()
