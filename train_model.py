"""
============================================================
  TRAIN MODEL - CNN KLASIFIKASI JENIS MOBIL
  Studi Kasus : Klasifikasi Jenis Mobil Berdasarkan Citra
  Dataset     : ademboukhris/cars-body-type-cropped (Kaggle)
  BAB 10      : Convolutional Neural Networks
============================================================
"""

# ============================================================
# 1. IMPORT LIBRARY
# ============================================================
import os, json, random, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix

# Reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 60)
print("  BAB 10 - Convolutional Neural Networks")
print("  Studi Kasus: Klasifikasi Jenis Mobil")
print("=" * 60)

# ============================================================
# 2. DOWNLOAD DATASET DARI KAGGLE
# ============================================================
print("\n[STEP 1] Download Dataset dari Kaggle...")

import kagglehub

# Gunakan path dari setup.py jika sudah ada, jika belum download otomatis
if os.path.exists("dataset_path.txt"):
    with open("dataset_path.txt") as f:
        path = f.read().strip()
    print(f"[INFO] Dataset ditemukan di: {path}")
else:
    print("[INFO] Mendownload dataset dari Kaggle...")
    path = kagglehub.dataset_download("ademboukhris/cars-body-type-cropped")
    with open("dataset_path.txt", "w") as f:
        f.write(path)
    print(f"[INFO] Dataset disimpan di: {path}")

print("Path to dataset files:", path)

# Temukan folder train/test/valid secara rekursif (berapa pun level dalamnya)
TRAIN_DIR = None
TEST_DIR  = None
VALID_DIR = None

for root, dirs, files in os.walk(path):
    bn = os.path.basename(root).lower()
    if bn in ('train', 'training') and TRAIN_DIR is None:
        TRAIN_DIR = root
    elif bn in ('test', 'testing') and TEST_DIR is None:
        TEST_DIR = root
    elif bn in ('val', 'valid', 'validation') and VALID_DIR is None:
        VALID_DIR = root

# Gunakan valid sebagai test jika tidak ada folder test
if TEST_DIR is None and VALID_DIR is not None:
    TEST_DIR = VALID_DIR

if TRAIN_DIR is None:
    TRAIN_DIR = path

print(f"\n[INFO] Train directory : {TRAIN_DIR}")
print(f"[INFO] Test  directory : {TEST_DIR}")

# ============================================================
# 3. CEK KELAS DAN JUMLAH DATA
# ============================================================
print("\n[STEP 2] Eksplorasi Dataset...")

CLASS_NAMES = sorted([
    d for d in os.listdir(TRAIN_DIR)
    if os.path.isdir(os.path.join(TRAIN_DIR, d))
])
NUM_CLASSES = len(CLASS_NAMES)
print(f"\n[INFO] Kelas yang ditemukan ({NUM_CLASSES}): {CLASS_NAMES}")

# Hitung gambar per kelas
class_counts = {}
for cls in CLASS_NAMES:
    path_cls = os.path.join(TRAIN_DIR, cls)
    imgs = [f for f in os.listdir(path_cls)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    class_counts[cls] = len(imgs)
    print(f"  {cls:20s}: {len(imgs):5d} gambar")

total = sum(class_counts.values())
print(f"\n  Total gambar : {total}")

# ============================================================
# 4. BUAT DATASET CSV
# ============================================================
print("\n[STEP 3] Membuat dataset CSV...")

os.makedirs('dataset', exist_ok=True)
rows = []
for cls in CLASS_NAMES:
    path_cls = os.path.join(TRAIN_DIR, cls)
    for f in os.listdir(path_cls):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            rows.append({'filename': f, 'label': cls,
                         'filepath': os.path.join(path_cls, f)})

df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
split_idx   = int(len(df) * 0.8)
df_train    = df.iloc[:split_idx][['filename', 'label']]
df_test     = df.iloc[split_idx:][['filename', 'label']]

df_train.to_csv('dataset/train.csv', index=False)
df_test.to_csv('dataset/test.csv',   index=False)
print(f"  [SAVED] dataset/train.csv  ({len(df_train)} baris)")
print(f"  [SAVED] dataset/test.csv   ({len(df_test)} baris)")

# ============================================================
# 5. VISUALISASI SAMPEL DATA
# ============================================================
print("\n[STEP 4] Visualisasi Sampel Gambar...")

os.makedirs('static/images', exist_ok=True)

samples_per_class = 4
fig, axes = plt.subplots(NUM_CLASSES, samples_per_class,
                          figsize=(samples_per_class * 3, NUM_CLASSES * 3))
fig.suptitle('Contoh Gambar per Kelas Jenis Mobil',
             fontsize=14, fontweight='bold')

for row_i, cls in enumerate(CLASS_NAMES):
    path_cls = os.path.join(TRAIN_DIR, cls)
    imgs     = [f for f in os.listdir(path_cls)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    sampled  = random.sample(imgs, min(samples_per_class, len(imgs)))

    for col_i in range(samples_per_class):
        ax = axes[row_i][col_i] if NUM_CLASSES > 1 else axes[col_i]
        if col_i < len(sampled):
            img = Image.open(os.path.join(path_cls, sampled[col_i])).convert('RGB')
            ax.imshow(img.resize((128, 128)))
            if col_i == 0:
                ax.set_ylabel(cls, fontsize=10, fontweight='bold', rotation=90, labelpad=8)
        else:
            ax.axis('off')
        ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig('static/images/sampel_data.png', dpi=120, bbox_inches='tight')
plt.close()
print("  [SAVED] static/images/sampel_data.png")

# Salin 1 contoh per kelas ke static
for cls in CLASS_NAMES:
    path_cls = os.path.join(TRAIN_DIR, cls)
    imgs = [f for f in os.listdir(path_cls)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if imgs:
        shutil.copy(os.path.join(path_cls, imgs[0]),
                    f'static/images/sample_{cls}.jpg')

# ============================================================
# 6. KONFIGURASI HYPERPARAMETER
# ============================================================
IMG_SIZE      = 128
BATCH_SIZE    = 32
EPOCHS        = 30
LEARNING_RATE = 0.001
VAL_SPLIT     = 0.2

print(f"\n[STEP 5] Konfigurasi Hyperparameter:")
print(f"  IMG_SIZE      : {IMG_SIZE} x {IMG_SIZE} piksel")
print(f"  BATCH_SIZE    : {BATCH_SIZE}")
print(f"  MAX EPOCHS    : {EPOCHS}")
print(f"  LEARNING_RATE : {LEARNING_RATE}")
print(f"  VAL_SPLIT     : {VAL_SPLIT * 100:.0f}%")

# ============================================================
# 7. DATA AUGMENTASI & GENERATOR
# ============================================================
print("\n[STEP 6] Augmentasi Data & ImageDataGenerator...")

# Generator TRAINING (dengan augmentasi)
train_datagen = ImageDataGenerator(
    rescale          = 1./255,
    rotation_range   = 25,
    width_shift_range= 0.2,
    height_shift_range= 0.2,
    shear_range      = 0.15,
    zoom_range       = 0.2,
    horizontal_flip  = True,
    brightness_range = [0.8, 1.2],
    fill_mode        = 'nearest',
    validation_split = VAL_SPLIT
)

# Generator TEST (hanya normalisasi, tanpa augmentasi)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size   = (IMG_SIZE, IMG_SIZE),
    batch_size    = BATCH_SIZE,
    class_mode    = 'categorical',
    subset        = 'training',
    shuffle       = True,
    seed          = SEED
)
val_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size   = (IMG_SIZE, IMG_SIZE),
    batch_size    = BATCH_SIZE,
    class_mode    = 'categorical',
    subset        = 'validation',
    shuffle       = False,
    seed          = SEED
)
eval_dir = TEST_DIR if (TEST_DIR and os.path.isdir(TEST_DIR)) else TRAIN_DIR
test_gen = test_datagen.flow_from_directory(
    eval_dir,
    target_size   = (IMG_SIZE, IMG_SIZE),
    batch_size    = BATCH_SIZE,
    class_mode    = 'categorical',
    shuffle       = False
)

# Update nama kelas dari generator (alphabetical)
CLASS_NAMES = list(train_gen.class_indices.keys())
NUM_CLASSES  = len(CLASS_NAMES)

print(f"\n[INFO] Kelas Final    : {CLASS_NAMES}")
print(f"[INFO] Data Training  : {train_gen.samples} gambar")
print(f"[INFO] Data Validasi  : {val_gen.samples} gambar")
print(f"[INFO] Data Test      : {test_gen.samples} gambar")

# ============================================================
# 8. MEMBANGUN ARSITEKTUR MODEL CNN
# ============================================================
print("\n[STEP 7] Membangun Arsitektur Model CNN...")
print("-" * 60)

model = Sequential([
    # ── BLOK KONVOLUSI 1 : Fitur Dasar (tepi, garis) ─────────
    Conv2D(32, (3, 3), activation='relu', padding='same',
           input_shape=(IMG_SIZE, IMG_SIZE, 3), name='conv1a'),
    BatchNormalization(name='bn1'),
    Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1b'),
    MaxPooling2D((2, 2), name='pool1'),
    Dropout(0.25, name='drop1'),

    # ── BLOK KONVOLUSI 2 : Pola Menengah (bentuk komponen) ───
    Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2a'),
    BatchNormalization(name='bn2'),
    Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2b'),
    MaxPooling2D((2, 2), name='pool2'),
    Dropout(0.25, name='drop2'),

    # ── BLOK KONVOLUSI 3 : Fitur Kompleks (siluet kendaraan) ─
    Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3a'),
    BatchNormalization(name='bn3'),
    Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3b'),
    MaxPooling2D((2, 2), name='pool3'),
    Dropout(0.30, name='drop3'),

    # ── BLOK KONVOLUSI 4 : Fitur Abstrak Tinggi ──────────────
    Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4a'),
    BatchNormalization(name='bn4'),
    MaxPooling2D((2, 2), name='pool4'),
    Dropout(0.30, name='drop4'),

    # ── FULLY CONNECTED LAYER ─────────────────────────────────
    Flatten(name='flatten'),
    Dense(512, activation='relu', name='fc1'),
    BatchNormalization(name='bn_fc'),
    Dropout(0.50, name='drop_fc1'),
    Dense(256, activation='relu', name='fc2'),
    Dropout(0.30, name='drop_fc2'),

    # ── OUTPUT LAYER ──────────────────────────────────────────
    Dense(NUM_CLASSES, activation='softmax', name='output')
], name='CNN_KlasifikasiMobil')

model.summary()
print("-" * 60)

# ============================================================
# 9. KOMPILASI MODEL
# ============================================================
print("\n[STEP 8] Kompilasi Model...")

model.compile(
    optimizer = Adam(learning_rate=LEARNING_RATE),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)
print(f"  Optimizer : Adam (learning_rate={LEARNING_RATE})")
print(f"  Loss      : categorical_crossentropy")
print(f"  Metrics   : accuracy")

# ============================================================
# 10. CALLBACK TRAINING
# ============================================================
print("\n[STEP 9] Konfigurasi Callback...")

os.makedirs('model', exist_ok=True)

callbacks = [
    # Hentikan training jika val_loss tidak membaik 8 epoch
    EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    # Kurangi learning rate jika val_loss stagnan 4 epoch
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    ),
    # Simpan model terbaik secara otomatis
    ModelCheckpoint(
        'model/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
]
print("  EarlyStopping      : patience=8, restore_best_weights=True")
print("  ReduceLROnPlateau  : factor=0.5, patience=4")
print("  ModelCheckpoint    : simpan model terbaik (val_accuracy)")

# ============================================================
# 11. TRAINING MODEL
# ============================================================
print("\n[STEP 10] Melatih Model CNN...")
print("=" * 60)

history = model.fit(
    train_gen,
    epochs          = EPOCHS,
    validation_data = val_gen,
    callbacks       = callbacks,
    verbose         = 1
)

epoch_run     = len(history.history['accuracy'])
best_val_acc  = max(history.history['val_accuracy'])
best_val_loss = min(history.history['val_loss'])

print("\n" + "=" * 60)
print(f"  Training selesai! ({epoch_run} dari {EPOCHS} epoch dijalankan)")
print(f"  Best Val Accuracy : {best_val_acc  * 100:.2f}%")
print(f"  Best Val Loss     : {best_val_loss:.4f}")
print("=" * 60)

# ============================================================
# 12. VISUALISASI GRAFIK TRAINING (EPOCH)
# ============================================================
print("\n[STEP 11] Membuat Grafik Akurasi & Loss per Epoch...")

ep = range(1, epoch_run + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f'Hasil Training CNN – Klasifikasi Jenis Mobil\n'
             f'({epoch_run} Epoch | Best Val Acc: {best_val_acc*100:.2f}%)',
             fontsize=13, fontweight='bold')

# -- Grafik Akurasi per Epoch
ax1.plot(ep, [v*100 for v in history.history['accuracy']],
         label='Train Accuracy', color='royalblue', lw=2, marker='o', ms=3)
ax1.plot(ep, [v*100 for v in history.history['val_accuracy']],
         label='Val Accuracy',   color='darkorange', lw=2, marker='s', ms=3)
ax1.axhline(y=best_val_acc*100, color='green', ls='--', alpha=0.6,
            label=f'Best Val: {best_val_acc*100:.1f}%')
ax1.set_title('Akurasi per Epoch', fontweight='bold')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Akurasi (%)')
ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_ylim(0, 105)
for ep_val, acc in enumerate(history.history['val_accuracy'], 1):
    if acc == best_val_acc:
        ax1.annotate(f'Best\n{acc*100:.1f}%', xy=(ep_val, acc*100),
                     xytext=(ep_val+0.5, acc*100-8),
                     fontsize=8, color='green',
                     arrowprops=dict(arrowstyle='->', color='green', lw=1))

# -- Grafik Loss per Epoch
ax2.plot(ep, history.history['loss'],
         label='Train Loss', color='royalblue', lw=2, marker='o', ms=3)
ax2.plot(ep, history.history['val_loss'],
         label='Val Loss',   color='darkorange', lw=2, marker='s', ms=3)
ax2.axhline(y=best_val_loss, color='green', ls='--', alpha=0.6,
            label=f'Best Loss: {best_val_loss:.4f}')
ax2.set_title('Loss per Epoch', fontweight='bold')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('static/images/training_history.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] static/images/training_history.png")

# -- Tabel ringkasan epoch
print("\n  Ringkasan per Epoch:")
print(f"  {'Epoch':>6} | {'Train Acc':>10} | {'Val Acc':>10} | {'Train Loss':>11} | {'Val Loss':>9}")
print("  " + "-" * 58)
for i in range(epoch_run):
    marker = " <<" if history.history['val_accuracy'][i] == best_val_acc else ""
    print(f"  {i+1:>6} | "
          f"{history.history['accuracy'][i]*100:>9.2f}% | "
          f"{history.history['val_accuracy'][i]*100:>9.2f}% | "
          f"{history.history['loss'][i]:>11.4f} | "
          f"{history.history['val_loss'][i]:>9.4f}{marker}")

# ============================================================
# 13. EVALUASI MODEL
# ============================================================
print("\n[STEP 12] Evaluasi Model pada Data Test...")
print("-" * 60)

test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"\n  Test Accuracy : {test_acc * 100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")

# ============================================================
# 14. CONFUSION MATRIX
# ============================================================
print("\n[STEP 13] Membuat Confusion Matrix...")

test_gen.reset()
y_pred_proba = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = test_gen.classes

cm = confusion_matrix(y_true, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Confusion Matrix – Klasifikasi Jenis Mobil', fontsize=13, fontweight='bold')

# Raw count
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[0], linewidths=0.5)
axes[0].set_title('Jumlah Prediksi'); axes[0].set_xlabel('Prediksi')
axes[0].set_ylabel('Label Asli'); axes[0].tick_params(axis='x', rotation=30)

# Normalized
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[1], linewidths=0.5)
axes[1].set_title('Proporsi (%)'); axes[1].set_xlabel('Prediksi')
axes[1].set_ylabel('Label Asli'); axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('static/images/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] static/images/confusion_matrix.png")

# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================
print("\n[STEP 14] Classification Report:")
print("=" * 60)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# -- Grafik Precision / Recall / F1 per kelas
report_dict = classification_report(
    y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
)
metrics_data = {
    'Precision': [report_dict[c]['precision'] for c in CLASS_NAMES],
    'Recall'   : [report_dict[c]['recall']    for c in CLASS_NAMES],
    'F1-Score' : [report_dict[c]['f1-score']  for c in CLASS_NAMES],
}
x_pos = np.arange(len(CLASS_NAMES))
width = 0.26
fig, ax = plt.subplots(figsize=(11, 5))
colors = ['#2196F3', '#4CAF50', '#FF9800']
for i, (metric, vals) in enumerate(metrics_data.items()):
    bars = ax.bar(x_pos + i*width, vals, width, label=metric,
                  color=colors[i], alpha=0.85, edgecolor='black', lw=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_title('Precision, Recall & F1-Score per Kelas', fontsize=12, fontweight='bold')
ax.set_xlabel('Kelas Jenis Mobil'); ax.set_ylabel('Nilai Metrik')
ax.set_xticks(x_pos + width); ax.set_xticklabels(CLASS_NAMES, rotation=20)
ax.set_ylim(0, 1.15); ax.legend(); ax.grid(axis='y', alpha=0.3)
ax.axhline(0.8, color='red', ls='--', alpha=0.4, label='Threshold 80%')
plt.tight_layout()
plt.savefig('static/images/metrics_kelas.png', dpi=150, bbox_inches='tight')
plt.close()
print("  [SAVED] static/images/metrics_kelas.png")

# ============================================================
# 16. SIMPAN MODEL .h5
# ============================================================
print("\n[STEP 15] Menyimpan Model...")

model.save('model/cnn_mobil.h5')
print("  [SAVED] model/cnn_mobil.h5")

# Simpan metadata model
meta = {
    'class_names'   : CLASS_NAMES,
    'num_classes'   : NUM_CLASSES,
    'img_size'      : IMG_SIZE,
    'test_accuracy' : round(float(test_acc),  4),
    'test_loss'     : round(float(test_loss), 4),
    'epochs_run'    : epoch_run,
    'best_val_acc'  : round(float(best_val_acc), 4),
    'best_val_loss' : round(float(best_val_loss), 4),
    'history'       : {
        'accuracy'    : [round(v, 4) for v in history.history['accuracy']],
        'val_accuracy': [round(v, 4) for v in history.history['val_accuracy']],
        'loss'        : [round(v, 4) for v in history.history['loss']],
        'val_loss'    : [round(v, 4) for v in history.history['val_loss']],
    }
}
with open('model/model_info.json', 'w') as f:
    json.dump(meta, f, indent=2)
print("  [SAVED] model/model_info.json")

# ============================================================
# 17. RINGKASAN AKHIR
# ============================================================
print("\n" + "=" * 60)
print("  RINGKASAN HASIL TRAINING")
print("=" * 60)
print(f"  Dataset          : ademboukhris/cars-body-type-cropped")
print(f"  Jumlah Kelas     : {NUM_CLASSES}  ({', '.join(CLASS_NAMES)})")
print(f"  Ukuran Input     : {IMG_SIZE} x {IMG_SIZE} piksel")
print(f"  Epoch Dijalankan : {epoch_run} dari {EPOCHS} epoch")
print(f"  Best Val Accuracy: {best_val_acc  * 100:.2f}%")
print(f"  Test Accuracy    : {test_acc * 100:.2f}%")
print(f"  Test Loss        : {test_loss:.4f}")
print("=" * 60)
print("\n  File Output:")
print("  model/cnn_mobil.h5             -> Model terlatih (.h5)")
print("  model/best_model.keras         -> Model terbaik (checkpoint)")
print("  model/model_info.json          -> Metadata & riwayat epoch")
print("  static/images/training_history.png -> Grafik akurasi & loss")
print("  static/images/confusion_matrix.png -> Confusion matrix")
print("  static/images/metrics_kelas.png    -> Precision/Recall/F1")
print("  dataset/train.csv              -> Data training (CSV)")
print("  dataset/test.csv               -> Data testing  (CSV)")
print("\n  Jalankan: python app.py  --> buka http://localhost:5000")
print("=" * 60)
