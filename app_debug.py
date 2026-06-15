import streamlit as st
print("STEP 1: Streamlit imported")

import torch
import open_clip
print("STEP 2: torch and open_clip imported")
import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
import os
print("STEP 3: all imports done")

QDRANT_PATH = r'C:\Users\Asus\embroidery-finder\qdrant_db'

print("STEP 4: starting st.title")
st.title("🧵 Krishna Embroidery Finder")
print("STEP 5: title rendered")

st.write("Debug test - if you see this, basic rendering works!")
print("STEP 6: write rendered")