# Recomendação por Similaridade de Imagens com DINOv2 e FAISS 🔗🎯
Olá, <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Hand%20gestures/Waving%20Hand.png" alt="Waving Hand" width="25" height="25" />
bem vindo ao meu projeto de Recomendação por Similaridade usando: 
- O dataset [E-commerce Product Images](https://www.kaggle.com/datasets/vikashrajluhaniwal/fashion-images);
- O modelo DINOv2 para extração de características das imagens;
- O FAISS para indexação, gerando uma busca mais rápida dentro do dataset;
- Poetry para organizar as dependências do projeto;
- Flask para criar uma interface que possibilita utilizar a extração de características e indexação feitas de maneira visual e intuitiva!
  
Vamos ver como ele foi desenvolvido!

## 🦖 DINOv2
DINOv2 (Self-supervised Vision Transformers) é um modelo de visão computacional desenvolvido pela Meta AI. Ele utiliza aprendizado auto-supervisionado para extrair representações visuais ricas a partir de imagens, sem a necessidade de rótulos manuais. O modelo foi treinado em um conjunto massivo de 142 milhões de imagens, permitindo que ele aprenda representações altamente generalizáveis e úteis para tarefas como segmentação, classificação e busca por similaridade. Lançado em abril de 2023, DINOv2 é amplamente utilizado em aplicações que exigem extração eficiente de características visuais.

## ⚡ FAISS
FAISS (Facebook AI Similarity Search) é uma biblioteca de código aberto desenvolvida pela Meta AI para busca eficiente de vetores de alta dimensão. Ele é projetado para lidar com grandes volumes de dados e permite encontrar os vetores mais similares em um índice de maneira extremamente rápida. Utilizando técnicas de quantização, normalização e busca aproximada, FAISS é amplamente utilizado em aplicações como sistemas de recomendação, recuperação de imagens e busca semântica em bases de dados massivas.

## 🚀 Tecnologias Utilizadas

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Torch](https://img.shields.io/badge/Torch-1.7.1%2B-red)
![Transformers](https://img.shields.io/badge/Transformers-4.3.0%2B-lightgrey)
![Pandas](https://img.shields.io/badge/Pandas-1.2.0%2B-yellowgreen)
![Faiss](https://img.shields.io/badge/Faiss-1.6.3%2B-brightgreen)
![Pillow](https://img.shields.io/badge/Pillow-8.1.0%2B-orange)
![Requests](https://img.shields.io/badge/Requests-2.25.1%2B-yellow)
![JSON](https://img.shields.io/badge/JSON-1.0-brightblue)
![Flask](https://img.shields.io/badge/Flask-Microframework-black?style=for-the-badge&logo=flask)


## 📌 Como funciona a extração e indexação de características

1. **Coleta das imagens:** As URLs das imagens são extraídas de um arquivo CSV (fashion.csv).

2. **Pré-processamento das imagens:**

- As imagens são baixadas das URLs.

- Convertidas para o formato RGB.

3. **Extração de características:**

- O modelo facebook/dinov2-small da Hugging Face é utilizado para extrair embeddings das imagens.

- Os embeddings representam características únicas das imagens.

4. **Indexação com Faiss:**

- Os embeddings são normalizados.

- São adicionados a um índice vetorial (faiss.IndexFlatL2).

5. **Armazenamento:**

- O índice é salvo no arquivo vector.index.

- Os identificadores das imagens são armazenados no image_ids.json.

O objetivo é permitir a busca eficiente por imagens similares utilizando a comparação de embeddings.

## 🎨🖥️ Frontend
![image](https://github.com/user-attachments/assets/b3852eef-8198-40d0-8836-784c187d8b5f)

O frontend foi criado com Flask para permitir a navegação do usuário pelo dataset. Para gerar uma recomendação por similaridade, basta clicar em uma das imagens e o sistema irá encontrar as imagens mais parecidas dentro do dataset.

![image](https://github.com/user-attachments/assets/7eb2684e-8872-4dbd-a78a-9a8e6d854e1e)

A rota `/get_similar` recebe a URL da imagem clicada e retorna uma lista de imagens similares com base na extração de embeddings usando `facebook/dinov2-small` e na busca vetorial com Faiss.


# Como usar
1. Certifique-se de usar uma versão do Python maior que a 3.8, mas recomendo a 3.12.6, que é a que usei para esse projeto
   - ```python --version```  
2. Crie um abmiente virtual python usando o comando abaixo, ou através da IDE que estiver usando
   - ```python -m venv .venv```
   - Certifique-se que a IDE está reconhecendo o interpretador da venv
3. Instale o poetry na venv e instale as dependências com o poetry
   - ```pip install poetry```
   - ```poetry install```
4. Execute o app.py que está na raiz do projeto

# 📚 Referências
- [CLIP Vs DINOv2 in image similarity](https://medium.com/aimonks/clip-vs-dinov2-in-image-similarity-6fa5aa7ed8c6)
- [Building a Visual Similarity-based Recommendation System Using Python](https://medium.com/geekculture/building-a-visual-similarity-based-recommendation-system-using-python-872a5bea568e)
- [Image similarity with DINOv2 and FAISS](https://medium.com/aimonks/image-similarity-with-dinov2-and-faiss-741744bc5804)
