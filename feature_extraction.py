import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import faiss
import time
import json  # Adicionado para salvar os nomes dos arquivos
import pandas as pd
import io
import requests

# Carregar o modelo e processador
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)

df = pd.read_csv("./dataset/fashion.csv")

image_urls = df["ImageURL"].tolist()

# Criar índice Faiss
index = faiss.IndexFlatL2(384)
ids = []  # Lista para armazenar IDs de imagens

t0 = time.time()
for idx, url in enumerate(image_urls):
    try:
        # Obter a imagem da URL
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        
        # Pré-processar e extrair embeddings
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt").to(device)
            outputs = model(**inputs)
        features = outputs.last_hidden_state
        
        # Normalizar e adicionar ao índice
        vector = features.mean(dim=1).detach().cpu().numpy().astype("float32")
        faiss.normalize_L2(vector)
        index.add(vector)
        ids.append(f"image_{idx}")  # Use um identificador para a imagem
    except Exception as e:
        print(f"Erro ao processar a URL {url}: {e}")

print("Extração finalizada em:", time.time() - t0)

# Salvamento do índice localmente
faiss.write_index(index, "vector.index")

with open("image_ids.json", "w") as f:
    json.dump(ids, f)