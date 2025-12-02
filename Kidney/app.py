from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from pathlib import Path
from werkzeug.utils import secure_filename
import os
from yolo_infer import detect_kidney_stones

# ----------------- CREATE APP FIRST -----------------
app = Flask(__name__)
app.secret_key = "dev-secret-change-me"

# ----------------- PATHS -----------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
ALLOWED_EXTS = {"png", "jpg", "jpeg", "dcm"}

app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- ROUTES -----------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Please choose an MRI image (PNG/JPG or DICOM .dcm).", "error")
            return redirect(url_for("index"))
        if not allowed_file(file.filename):
            flash("Unsupported file type.", "error")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        upload_path = UPLOAD_DIR / filename
        file.save(upload_path)

        try:
            annotated_path = detect_kidney_stones(str(upload_path), str(OUTPUT_DIR))
        except Exception as e:
            flash(f"Detection failed: {e}", "error")
            return redirect(url_for("index"))

        return render_template(
            "result.html",
            uploaded_image_url=f"/uploads/{filename}",
            result_image_url=f"/static/outputs/{Path(annotated_path).name}",
            detection_stats=["Detected kidney stones (example)"]
        )

    return render_template("index.html")

# ----------------- RUN APP -----------------
if __name__ == "__main__":
    app.run(debug=True)
