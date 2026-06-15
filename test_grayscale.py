import torch
import open_clip
from PIL import Image, ImageOps
import cv2
import numpy as np

device = "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
model.eval()
model = model.to(device)

CUSTOMER_IMG = r'C:\Users\Asus\embroidery-finder\test_customer.jpg'
DB_IMG = r'C:\Users\Asus\embroidery-finder\dst_images\Varsha Creations_2660 design_2660 N-Full.png'  # adjust filename if needed

def get_embedding(pil_image):
    image_tensor = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features

def to_grayscale_3ch(pil_image):
    gray = ImageOps.grayscale(pil_image)
    return Image.merge("RGB", (gray, gray, gray))

def to_edges_3ch(pil_image):
    img_array = np.array(pil_image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

# Load images
customer = Image.open(CUSTOMER_IMG).convert('RGB')
db_image = Image.open(DB_IMG).convert('RGB')

print("=" * 50)
print("TEST 1: Original color images")
e1 = get_embedding(customer)
e2 = get_embedding(db_image)
sim = torch.nn.functional.cosine_similarity(e1, e2).item()
print(f"Similarity: {sim*100:.2f}%")

print("=" * 50)
print("TEST 2: Grayscale images")
e1g = get_embedding(to_grayscale_3ch(customer))
e2g = get_embedding(to_grayscale_3ch(db_image))
simg = torch.nn.functional.cosine_similarity(e1g, e2g).item()
print(f"Similarity: {simg*100:.2f}%")

print("=" * 50)
print("TEST 3: Edge-detected images")
e1e = get_embedding(to_edges_3ch(customer))
e2e = get_embedding(to_edges_3ch(db_image))
sime = torch.nn.functional.cosine_similarity(e1e, e2e).item()
print(f"Similarity: {sime*100:.2f}%")

print("=" * 50)
print("Comparison: original color score vs grayscale vs edges")
print(f"Color:     {sim*100:.2f}%")
print(f"Grayscale: {simg*100:.2f}%")
print(f"Edges:     {sime*100:.2f}%")