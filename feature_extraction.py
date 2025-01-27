import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import faiss
import numpy as np
import os
import time
import json  # Adicionado para salvar os nomes dos arquivos

# Carregar o modelo e processador
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)

# Listar todas as imagens no diretório do dataset
images = []
for root, dirs, files in os.walk("./dataset"):
    for file in files:
        if file.endswith("jpg"):
            images.append(os.path.join(root, file))


# Define uma função que normaliza embeddings e adiciona ao índice
def add_vector_to_index(embedding, index, ids):
    vector = embedding.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)
    index.add(vector)
    ids.append(image_id)


# Criar o índice Faiss usando o tipo FlatL2 com 384 dimensões
index = faiss.IndexFlatL2(384)
ids = []

t0 = time.time()
for image_path in images:
    img = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        outputs = model(**inputs)
    features = outputs.last_hidden_state
    image_id = os.path.basename(image_path).split(".")[
        0
    ]  # Extraindo o ID da imagem do nome do arquivo
    add_vector_to_index(features.mean(dim=1), index, ids)

print("Extração finalizada em:", time.time() - t0)

# Salvamento do índice localmente
faiss.write_index(index, "vector.index")

# Salvamento dos nomes dos arquivos (IDs)
with open("image_ids.json", "w") as f:
    json.dump(ids, f)
