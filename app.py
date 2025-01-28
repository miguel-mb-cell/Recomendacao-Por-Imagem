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

# Carrega as URLs do dataset
df = pd.read_csv('./dataset/fashion.csv')  # Substitua pelo caminho correto do CSV
IMAGES = df['ImageURL'].tolist()

@app.route("/", methods=["GET"])
def index():
    page = int(request.args.get("page", 1))  # Página atual
    apparel = request.args.get("apparel")  # Filtro para roupas
    footwear = request.args.get("footwear")  # Filtro para sapatos

    # Filtrar o DataFrame
    filtered_df = df
    if apparel:
        filtered_df = filtered_df[(filtered_df['Category'] == 'Apparel') & (filtered_df['Gender'] == apparel)]
    elif footwear:
        filtered_df = filtered_df[(filtered_df['Category'] == 'Footwear') & (filtered_df['Gender'] == footwear)]

    # Paginação
    per_page = 6      # Número de imagens por página
    total_pages = (len(filtered_df) + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page

    # Obter URLs das imagens paginadas
    images = filtered_df['ImageURL'].iloc[start:end].tolist()

    return render_template("index.html", images=images, page=page, total_pages=total_pages, apparel=apparel, footwear=footwear)

# Carregar o índice faiss e os IDs das imagens
faiss_index = faiss.read_index("vector.index")
with open("image_ids.json", "r") as f:
    image_ids = json.load(f)

# Carregar o modelo e processador
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)

@app.route('/get_similar', methods=['POST'])
def get_similar():
    data = request.get_json()
    image_path = data.get("image_path")

    if not image_path:
        return jsonify({"error": "No image path provided"}), 400

    # Verificar se a URL é válida
    if not (image_path.startswith("http://") or image_path.startswith("https://")):
        return jsonify({"error": "Invalid URL provided"}), 400

    try:
        # Baixar a imagem da URL
        response = requests.get(image_path, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        # Processar a imagem
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)

        # Gerar vetor normalizado
        embeddings = outputs.last_hidden_state.mean(dim=1)
        vector = embeddings.detach().cpu().numpy().astype("float32")
        vector = vector.reshape(1, -1)
        faiss.normalize_L2(vector)

        # Buscar no índice FAISS
        d, i = faiss_index.search(vector, 3)  # Busca 3 similares
        indices_found = i[0]

        # Recuperar URLs das imagens semelhantes
        img_paths = [IMAGES[idx] for idx in indices_found if idx < len(IMAGES)]

        return jsonify({"similar_images": img_paths})
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
