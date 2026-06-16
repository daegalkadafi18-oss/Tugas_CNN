# 🚗 CNN Klasifikasi Jenis Mobil
> Tugas BAB 10 – Convolutional Neural Networks  
> Dataset: [Cars Body Type Cropped](https://www.kaggle.com/datasets/ademboukhris/cars-body-type-cropped) (Kaggle)

---

## 📁 Struktur Project

```
Project_CNN_Mobil/
├── dataset/
│   ├── train.csv            # Dibuat otomatis saat training
│   └── test.csv             # Dibuat otomatis saat training
├── model/
│   ├── cnn_mobil.h5         # Model CNN terlatih
│   ├── best_model.keras     # Checkpoint model terbaik
│   └── model_info.json      # Metadata & riwayat epoch
├── static/
│   ├── images/              # Grafik hasil training
│   └── uploads/             # Foto yang diupload user
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── result.html
│   ├── dataset.html
│   ├── epoch.html
│   └── about.html
├── app.py                   # Flask web application
├── train_model.py           # Script training CNN
├── setup.py                 # Install library + download dataset
├── requirements.txt
├── Procfile
├── nixpacks.toml
└── README.md
```

---

## ▶️ Cara Menjalankan (3 Langkah Saja)

### Langkah 1 — Buka Terminal di VS Code
```
Ctrl + ` (backtick)
```

### Langkah 2 — Buat & Aktifkan Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### Langkah 3 — Jalankan Setup (Install + Download Dataset Otomatis)
```bash
python setup.py
```
Setup akan:
- ✅ Install semua library dari `requirements.txt`
- ✅ Minta Kaggle Username & API Key (sekali saja)
- ✅ Download dataset otomatis dari Kaggle
- ✅ Simpan path dataset ke `dataset_path.txt`

### Langkah 4 — Training Model CNN
```bash
python train_model.py
```
Proses sekitar 10–20 menit. Hasil:
- `model/cnn_mobil.h5`
- `static/images/training_history.png`
- `static/images/confusion_matrix.png`
- `dataset/train.csv` & `dataset/test.csv`

### Langkah 5 — Jalankan Web App
```bash
python app.py
```

### Langkah 6 — Buka Browser
```
http://localhost:5000
```

---

## 🌐 Halaman Web

| Menu | URL | Isi |
|------|-----|-----|
| Beranda | `/` | Info model & statistik |
| Prediksi | `/predict` | Upload foto mobil |
| Dataset | `/dataset` | Tabel train.csv & test.csv |
| Epoch | `/epoch` | Riwayat training per epoch |
| Tentang | `/about` | Arsitektur CNN & info proyek |

---

## 🚀 Deploy ke Railway

```bash
# 1. Push ke GitHub
git init
git add .
git commit -m "CNN Klasifikasi Mobil"
git branch -M main
git remote add origin https://github.com/username/repo.git
git push -u origin main

# 2. Buka railway.app → New Project → Deploy from GitHub
# 3. Set Variables: KAGGLE_USERNAME dan KAGGLE_KEY
# 4. Generate Domain → selesai
```

---

## 🧠 Teknologi

- **TensorFlow / Keras** — Deep Learning CNN
- **Flask** — Web Framework
- **kagglehub** — Download dataset otomatis
- **Bootstrap 5** — Tampilan antarmuka
- **Railway** — Deployment platform
