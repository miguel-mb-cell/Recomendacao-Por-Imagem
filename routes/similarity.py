import faiss
import json
import torch
from transformers import AutoImageProcessor, AutoModel
from flask import Blueprint, request, jsonify
import requests
import io
from PIL import Image
from config import FAISS_INDEX_PATH, IMAGE_IDS_PATH
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar FAISS
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
with open(IMAGE_IDS_PATH, "r") as f:
    image_ids = json.load(f)

# Carregar modelo
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)

similarity_bp = Blueprint("similarity", __name__)

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "../dataset/fashion.csv")
df = pd.read_csv(csv_path)
IMAGES = df["ImageURL"].tolist()

@similarity_bp.route("/get_similar", methods=["POST"])
def get_similar():
    data = request.get_json()
    image_path = data.get("image_path")

    if not image_path:
        return jsonify({"error": "No image path provided"}), 400

    if not (image_path.startswith("http://") or image_path.startswith("https://")):
        return jsonify({"error": "Invalid URL provided"}), 400

    try:
        response = requests.get(image_path, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to("cpu")
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1)
        vector = embeddings.detach().cpu().numpy().astype("float32")
        vector = vector.reshape(1, -1)
        faiss.normalize_L2(vector)

        d, i = faiss_index.search(vector, 10)
        indices_and_distances = list(zip(i[0], d[0]))

        filtered_results = []
        seen_distances = set()

        for idx, dist in indices_and_distances:
            if idx < len(IMAGES) and dist > 0.0 and dist not in seen_distances:
                filtered_results.append((idx, dist))
                seen_distances.add(dist)

        top_results = sorted(filtered_results, key=lambda x: x[1])[:3]
        img_paths = [IMAGES[idx] for idx, _ in top_results]

        return jsonify({"similar_images": img_paths})
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500
