import json
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# Muat model dan metadata sekali saat aplikasi dijalankan
model = joblib.load(os.path.join(BASE, "model", "fish_model.joblib"))
with open(os.path.join(BASE, "model", "metadata.json")) as f:
    meta = json.load(f)


@app.route("/", methods=["GET", "POST"])
def index():
    hasil = None
    error = None
    peringatan = None
    nilai_input = ""

    if request.method == "POST":
        nilai_input = request.form.get("panjang", "").strip().replace(",", ".")
        try:
            panjang = float(nilai_input)
        except ValueError:
            panjang = None
            error = "Masukkan panjang ikan berupa angka, misalnya 30 atau 25.5."

        if panjang is not None:
            if not (meta["min_length"] <= panjang <= meta["max_length"]):
                error = (
                    f"Panjang harus antara {meta['min_length']:.1f} dan "
                    f"{meta['max_length']:.1f} cm (rentang data latih)."
                )
            else:
                X = pd.DataFrame([[panjang]], columns=[meta["feature"]])
                pred = float(model.predict(X)[0])
                if pred <= 0:
                    pred = 0.0
                    peringatan = (
                        "Model linear menghasilkan nilai di bawah nol untuk ikan sekecil ini, "
                        "sehingga estimasi tidak andal. Persamaan linear kurang cocok untuk "
                        f"panjang di bawah sekitar {meta['zero_weight_length']:.1f} cm."
                    )
                hasil = {"panjang": panjang, "berat": pred}

    return render_template(
        "index.html", hasil=hasil, error=error, peringatan=peringatan, meta=meta, nilai_input=nilai_input
    )


if __name__ == "__main__":
    app.run(debug=True)
