# Prediksi Berat Ikan Berdasarkan Panjang Ikan (Regresi Linear Sederhana + Flask)

Tugas Akhir 02 - Praktikum Pembelajaran Mesin.

- **Dataset:** Fish Market (Aung Pyae) - https://www.kaggle.com/datasets/aungpyaeap/fish-market
- **X (independen):** `Length3` (panjang ikan, cm)
- **y (dependen):** `Weight` (berat ikan, gram)
- **Model:** `LinearRegression` (scikit-learn), disimpan dengan Joblib

## Struktur
```
Fish.csv               dataset
train_model.py         EDA, pelatihan, evaluasi, simpan model
TA02_Fish_Notebook.ipynb   notebook untuk Colab/Jupyter
app.py                 aplikasi Flask
templates/index.html   antarmuka (Bootstrap)
model/                 fish_model.joblib + metadata.json
static/img/            grafik hasil EDA dan evaluasi
requirements.txt, Procfile, .python-version
```

## Menjalankan lokal
```bash
pip install -r requirements.txt
python train_model.py     # opsional, model sudah disertakan
python app.py             # buka http://127.0.0.1:5000
```

## Deploy ke Render (gratis)
1. Upload semua file ke repositori GitHub.
2. Render -> New -> Web Service -> pilih repositori.
3. Runtime: Python 3. Build Command: `pip install -r requirements.txt && python train_model.py`
4. Start Command: `gunicorn app:app`
5. Pilih instance Free, lalu Deploy. Salin URL aplikasi ke laporan.

Build Command memanggil `train_model.py` agar model dilatih ulang dengan versi scikit-learn yang sama di server.
