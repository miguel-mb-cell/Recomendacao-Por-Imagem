from flask import Flask
from routes.main import main_bp
from routes.similarity import similarity_bp
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)

app.register_blueprint(main_bp)
app.register_blueprint(similarity_bp)

if __name__ == "__main__":
    app.run()
