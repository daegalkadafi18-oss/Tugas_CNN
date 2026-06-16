"""
============================================================
  APP.PY - FLASK WEB APPLICATION
  CNN Klasifikasi Jenis Mobil Berdasarkan Citra Kendaraan
  BAB 10 - Convolutional Neural Networks
============================================================
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from werkzeug.utils import secure_filename
from PIL import Image

# ── Inisialisasi Flask ───────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'cnn_mobil_secret_2025'

# ── Konfigurasi Upload ───────────────────────────────────────
UPLOAD_FOLDER   = 'static/uploads'
ALLOWED_EXT     = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
app.config['UPLOAD_FOLDER']   = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024   # 10 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Load Model & Metadata ────────────────────────────────────
MODEL_PATH = 'model/cnn_mobil.h5'
INFO_PATH  = 'model/model_info.json'

model       = None
CLASS_NAMES = []
IMG_SIZE    = 128
MODEL_INFO  = {}

def load_resources():
    global model, CLASS_NAMES, IMG_SIZE, MODEL_INFO
    if os.path.exists(MODEL_PATH) and os.path.exists(INFO_PATH):
        model = load_model(MODEL_PATH)
        with open(INFO_PATH) as f:
            MODEL_INFO  = json.load(f)
        CLASS_NAMES = MODEL_INFO.get('class_names', [])
        IMG_SIZE    = MODEL_INFO.get('img_size', 128)
        print(f"[OK] Model loaded | Kelas: {CLASS_NAMES}")
    else:
        print("[WARN] Model belum ada. Jalankan train_model.py terlebih dahulu.")

load_resources()

# ── Deskripsi per kelas ──────────────────────────────────────
CLASS_DESC = {
    'Sedan'     : 'Mobil sedan berbadan rendah dengan desain 3-box (mesin, kabin, bagasi terpisah). Cocok untuk perjalanan kota dan jalan tol.',
    'SUV'       : 'Sport Utility Vehicle – kendaraan bertubuh tinggi dengan ground clearance besar, cocok untuk berbagai medan jalan.',
    'Hatchback' : 'Mobil kompak dengan pintu belakang terangkat (liftback). Efisien, lincah, dan hemat BBM untuk dalam kota.',
    'Truck'     : 'Kendaraan niaga dengan bak terbuka di bagian belakang, digunakan untuk angkutan barang atau kebutuhan komersial.',
    'Convertible': 'Mobil dengan atap yang bisa dibuka/ditutup (retractable roof). Memberikan pengalaman berkendara terbuka.',
    'Coupe'     : 'Mobil sporty dengan 2 pintu, desain atap melandai ke belakang, dan kabin yang lebih rendah.',
    'Minivan'   : 'Kendaraan keluarga besar dengan kapasitas penumpang tinggi dan ruang kabin yang luas.',
    'Wagon'     : 'Estate/station wagon – sedan dengan bagasi diperluas ke belakang tanpa pemisah kabin-bagasi.',
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def predict_image(img_path):
    """Prediksi jenis mobil dari path gambar."""
    img      = keras_image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr      = keras_image.img_to_array(img) / 255.0
    arr      = np.expand_dims(arr, axis=0)
    preds    = model.predict(arr, verbose=0)[0]
    idx      = int(np.argmax(preds))
    label    = CLASS_NAMES[idx]
    conf     = float(preds[idx]) * 100
    all_conf = {CLASS_NAMES[i]: round(float(preds[i]) * 100, 2)
                for i in range(len(CLASS_NAMES))}
    # Sort by confidence descending
    all_conf = dict(sorted(all_conf.items(), key=lambda x: x[1], reverse=True))
    desc     = CLASS_DESC.get(label, 'Jenis kendaraan terklasifikasi oleh model CNN.')
    return label, conf, all_conf, desc

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    """Halaman utama / beranda."""
    return render_template('index.html',
                           model_ready=(model is not None),
                           model_info=MODEL_INFO,
                           class_names=CLASS_NAMES)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Halaman prediksi – upload gambar & tampilkan hasil."""
    if request.method == 'GET':
        return render_template('predict.html', model_ready=(model is not None))

    # POST: terima file
    if 'file' not in request.files:
        flash('Tidak ada file yang dipilih!', 'danger')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('Pilih gambar terlebih dahulu.', 'warning')
        return redirect(request.url)

    if not allowed_file(file.filename):
        flash('Format file tidak didukung. Gunakan JPG / PNG / BMP.', 'danger')
        return redirect(request.url)

    if model is None:
        flash('Model belum tersedia. Jalankan train_model.py terlebih dahulu!', 'danger')
        return redirect(request.url)

    # Simpan & prediksi
    filename   = secure_filename(file.filename)
    save_path  = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    try:
        label, conf, all_conf, desc = predict_image(save_path)
    except Exception as e:
        flash(f'Terjadi error saat prediksi: {str(e)}', 'danger')
        return redirect(request.url)

    img_url = url_for('static', filename=f'uploads/{filename}')
    return render_template('result.html',
                           img_url=img_url,
                           label=label,
                           confidence=round(conf, 2),
                           all_conf=all_conf,
                           description=desc,
                           filename=filename)


@app.route('/about')
def about():
    """Halaman tentang proyek dan arsitektur CNN."""
    return render_template('about.html',
                           model_info=MODEL_INFO,
                           class_names=CLASS_NAMES)


@app.route('/dataset')
def dataset():
    """Halaman dataset – tampilkan train.csv dan test.csv."""
    train_data, test_data = [], []
    train_stats, test_stats = {}, {}
    train_total, test_total = 0, 0
    error_msg = None

    try:
        # ── Load train.csv ────────────────────────────────────
        if os.path.exists('dataset/train.csv'):
            df_train    = pd.read_csv('dataset/train.csv')
            train_data  = df_train.to_dict(orient='records')
            train_total = len(df_train)
            train_stats = df_train['label'].value_counts().to_dict()
        else:
            error_msg = "dataset/train.csv belum ada. Jalankan train_model.py terlebih dahulu."

        # ── Load test.csv ─────────────────────────────────────
        if os.path.exists('dataset/test.csv'):
            df_test    = pd.read_csv('dataset/test.csv')
            test_data  = df_test.to_dict(orient='records')
            test_total = len(df_test)
            test_stats = df_test['label'].value_counts().to_dict()

    except Exception as e:
        error_msg = f"Error membaca dataset: {str(e)}"

    return render_template('dataset.html',
                           train_data=train_data,
                           test_data=test_data,
                           train_stats=train_stats,
                           test_stats=test_stats,
                           train_total=train_total,
                           test_total=test_total,
                           class_names=CLASS_NAMES,
                           model_ready=(model is not None),
                           error_msg=error_msg)


@app.route('/epoch')
def epoch():
    """Halaman riwayat epoch training."""
    epoch_data = []
    error_msg  = None

    if MODEL_INFO and 'history' in MODEL_INFO:
        h = MODEL_INFO['history']
        for i in range(MODEL_INFO.get('epochs_run', 0)):
            epoch_data.append({
                'epoch'       : i + 1,
                'accuracy'    : round(h['accuracy'][i]     * 100, 2),
                'val_accuracy': round(h['val_accuracy'][i] * 100, 2),
                'loss'        : round(h['loss'][i],     4),
                'val_loss'    : round(h['val_loss'][i], 4),
                'is_best'     : h['val_accuracy'][i] == max(h['val_accuracy'])
            })
    else:
        error_msg = "Data epoch belum tersedia. Jalankan train_model.py terlebih dahulu."

    return render_template('epoch.html',
                           epoch_data=epoch_data,
                           model_info=MODEL_INFO,
                           class_names=CLASS_NAMES,
                           model_ready=(model is not None),
                           error_msg=error_msg)


# ── Entry point ──────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
