import cv2
import numpy as np
from PIL import Image

def to_edges_3ch(pil_image):
    img_array = np.array(pil_image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

customer = Image.open(r'C:\Users\Asus\embroidery-finder\test_customer.jpg').convert('RGB')
customer_edges = to_edges_3ch(customer)
customer_edges.save(r'C:\Users\Asus\embroidery-finder\edge_customer.png')

img_2660_full = Image.open(r'C:\Users\Asus\embroidery-finder\dst_images\Varsha Creations_2660 design_2660 N-Full.png').convert('RGB')
edges_2660_full = to_edges_3ch(img_2660_full)
edges_2660_full.save(r'C:\Users\Asus\embroidery-finder\edge_2660_full.png')

img_2660_alt = Image.open(r'C:\Users\Asus\embroidery-finder\dst_images\Varsha Creations_2660 design_5-6-11.png').convert('RGB')
edges_2660_alt = to_edges_3ch(img_2660_alt)
edges_2660_alt.save(r'C:\Users\Asus\embroidery-finder\edge_2660_alt.png')

img_3843 = Image.open(r'C:\Users\Asus\embroidery-finder\dst_images\Varsha Creations_SVC 3843_3843 Bu-1.png').convert('RGB')
edges_3843 = to_edges_3ch(img_3843)
edges_3843.save(r'C:\Users\Asus\embroidery-finder\edge_3843.png')

print("Saved 4 edge images")