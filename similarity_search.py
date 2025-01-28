import faiss
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import os
import json
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Carregar o índice faiss e os IDs das imagens
index = faiss.read_index("vector.index")
with open("image_ids.json", "r") as f:
    image_ids = json.load(f)

# Carregar imagem de entrada
image = Image.open("./teste.jpg")  # Utilize qualquer imagem fornecida pelo usuário

# Carregar o modelo e processador
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)

# Extrair características
with torch.no_grad():
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

# Normalizar características para a busca
embeddings = outputs.last_hidden_state.mean(dim=1)
vector = embeddings.detach().cpu().numpy().astype("float32")
vector = vector.reshape(1, -1)
faiss.normalize_L2(vector)

# Realizar busca no índice
d, i = index.search(vector, 3)
print("Distanças:", d, "Índices:", i)

# Recuperar os caminhos das imagens encontradas
indices_found = i[0]
img_paths = [
    f"./dataset/Apparel/Boys/Images/images_with_product_ids/{image_ids[idx]}.jpg"
    for idx in indices_found
]

# Verificação de existência dos arquivos
existing_img_paths = [
    (img_path, image_ids[idx])
    for img_path, idx in zip(img_paths, indices_found)
    if os.path.exists(img_path)
]


# Função para plotar imagens
def plot_images(img_paths, titles):
    plt.figure(figsize=(15, 5))
    for idx, (img_path, title) in enumerate(zip(img_paths, titles)):
        plt.subplot(1, len(img_paths), idx + 1)
        img = Image.open(img_path)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")
    plt.show()


# Plotar as imagens encontradas após verificação
if existing_img_paths:
    existing_img_paths, existing_titles = zip(*existing_img_paths)
    print(f"Recomendações baseadas em similaridade: {existing_titles}")
    plot_images(existing_img_paths, [f"Produto {title}" for title in existing_titles])
else:
    print("Nenhuma similaridade encontrada.")
