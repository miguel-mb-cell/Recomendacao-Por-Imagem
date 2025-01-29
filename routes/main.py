from flask import Blueprint, render_template, request
import pandas as pd
from config import DATASET_PATH

main_bp = Blueprint("main", __name__)

df = pd.read_csv(DATASET_PATH)

@main_bp.route("/", methods=["GET"])
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
