from flask import Flask, render_template, request, jsonify
import pandas as pd
import faiss
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import json
import os
import requests
import io

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)

df = pd.read_csv("./dataset/fashion.csv")
IMAGES = df["ImageURL"].tolist()


@app.route("/", methods=["GET"])
def index():
    page = int(request.args.get("page", 1))
    apparel = request.args.get("apparel")
    footwear = request.args.get("footwear")

    filtered_df = df
    if apparel:
        filtered_df = filtered_df[
            (filtered_df["Category"] == "Apparel") & (filtered_df["Gender"] == apparel)
        ]
    elif footwear:
        filtered_df = filtered_df[
            (filtered_df["Category"] == "Footwear")
            & (filtered_df["Gender"] == footwear)
        ]

    per_page = 6
    total_pages = (len(filtered_df) + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page

    images = filtered_df["ImageURL"].iloc[start:end].tolist()

    return render_template(
        "index.html",
        images=images,
        page=page,
        total_pages=total_pages,
        apparel=apparel,
        footwear=footwear,
    )


faiss_index = faiss.read_index("vector.index")
with open("image_ids.json", "r") as f:
    image_ids = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)


@app.route("/get_similar", methods=["POST"])
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
            inputs = processor(images=image, return_tensors="pt").to(device)
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


if __name__ == "__main__":
    app.run(debug=True)
