"""
============================================================
  SETUP.PY - Install Library + Download Dataset Otomatis
  Jalankan SEKALI sebelum training:
  >> python setup.py
============================================================
"""

import subprocess
import sys
import os

# ============================================================
# 1. INSTALL SEMUA LIBRARY DARI requirements.txt
# ============================================================
print("=" * 60)
print("  STEP 1: Install Library dari requirements.txt")
print("=" * 60)

subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "-r", "requirements.txt"
])
print("\n✅ Semua library berhasil diinstall!\n")

# ============================================================
# 2. DOWNLOAD DATASET DARI KAGGLE
# ============================================================
print("=" * 60)
print("  STEP 2: Download Dataset dari Kaggle")
print("=" * 60)
print("  Dataset : ademboukhris/cars-body-type-cropped")
print("  Harap tunggu...\n")

import kagglehub

# Download latest version
path = kagglehub.dataset_download("ademboukhris/cars-body-type-cropped")
print("Path to dataset files:", path)

# Simpan path ke file agar train_model.py bisa baca
with open("dataset_path.txt", "w") as f:
    f.write(path)

print(f"\n✅ Dataset berhasil didownload!")
print(f"   Path tersimpan di: dataset_path.txt")

# ============================================================
# 3. CEK ISI DATASET
# ============================================================
print("\n" + "=" * 60)
print("  STEP 3: Cek Isi Dataset")
print("=" * 60)

# Cari folder train/test/valid secara rekursif (berapa pun level dalamnya)
total_images = 0
found_dirs = {}

for root, dirs, files in os.walk(path):
    folder = os.path.basename(root).lower()
    imgs = [f for f in files
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if folder in ('train', 'training', 'test', 'testing', 'valid', 'validation'):
        found_dirs[root] = imgs
    # Hitung semua gambar di subfolder kelas
    parent = os.path.basename(os.path.dirname(root)).lower()
    if parent in ('train', 'training', 'test', 'testing', 'valid', 'validation'):
        total_images += len(imgs)
        rel = root.replace(path, "").lstrip(os.sep)
        print(f"  🖼  {rel:40s} → {len(imgs)} gambar")

print(f"\n  Total gambar ditemukan: {total_images}")

# Tampilkan path split yang ditemukan
print("\n  Folder split yang ditemukan:")
for d in found_dirs:
    rel = d.replace(path, "").lstrip(os.sep)
    print(f"  ✅ {rel}")

# ============================================================
# SELESAI
# ============================================================
print("\n" + "=" * 60)
print("  ✅ SETUP SELESAI!")
print("=" * 60)
print("""
  Langkah selanjutnya:

  1. Jalankan training model:
     >> python train_model.py

  2. Setelah training selesai, jalankan web app:
     >> python app.py

  3. Buka browser:
     >> http://localhost:5000
""")
