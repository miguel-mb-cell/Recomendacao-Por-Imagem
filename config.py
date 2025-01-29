import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "fashion.csv")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "vector.index")
IMAGE_IDS_PATH = os.path.join(BASE_DIR, "image_ids.json")
