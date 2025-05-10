import os
from flask import Flask, request, render_template
from blip_fast import describe_image
from serp import search_all_platforms  # Keep this, since google_search returns results for just 1 site

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    description = ""
    results = []
    if request.method == "POST":
        if "image" in request.files:
            img = request.files["image"]
            img_path = os.path.join(UPLOAD_FOLDER, img.filename)
            img.save(img_path)

            # 1. Get description from image
            description = describe_image(img_path)

            # 2. Search products using full platform scan
            results = search_all_platforms(description)

    return render_template("index.html", description=description, results=results)

if __name__ == "__main__":
    app.run(debug=True)
