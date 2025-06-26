import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os
from PIL import Image

# Import kelas Config dari file config.py (Import absolut yang benar)
from config import Config

# --- Variabel Global untuk Model dan Mapping Kelas ---
# Ini akan menyimpan instance model TensorFlow/Keras dan mapping kelas
# setelah dimuat pertama kali. Tujuannya agar tidak perlu memuat ulang
# setiap kali ada permintaan prediksi, yang akan sangat lambat.
_model = None
_idx_to_class = None

# --- Fungsi untuk Memuat Model dan Mapping Kelas ---
def load_model_and_class_indices():
    """
    Memuat model Keras dan mapping kelas ke memori.
    Fungsi ini dipanggil saat modul ini pertama kali diimpor,
    memastikan model siap digunakan sejak aplikasi dimulai.
    """
    global _model, _idx_to_class

    # Muat model hanya jika belum dimuat sebelumnya
    if _model is None:
        try:
            _model = tf.keras.models.load_model(Config.MODEL_PATH)
            print(f"Model AI berhasil dimuat dari: {Config.MODEL_PATH}")
            # Opsional: pesan bahwa model dimuat ke CPU/GPU
            if tf.config.list_physical_devices('GPU'):
                print("Model dimuat untuk inferensi GPU.")
            else:
                print("Model dimuat untuk inferensi CPU.")
        except Exception as e:
            print(f"ERROR: Gagal memuat model dari {Config.MODEL_PATH}. Pastikan file ada dan benar. Error: {e}")
            _model = None # Set ke None jika gagal, untuk penanganan error lebih lanjut

    # Muat mapping kelas hanya jika belum dimuat sebelumnya
    if _idx_to_class is None:
        try:
            with open(Config.CLASS_INDICES_PATH, 'r') as f:
                class_indices_raw = json.load(f)
            # Kunci dari JSON biasanya string, konversi ke integer karena indeks kelas adalah integer
            _idx_to_class = {int(k): v for k, v in class_indices_raw.items()}
            print(f"Mapping kelas berhasil dimuat dari: {Config.CLASS_INDICES_PATH}")
        except Exception as e:
            print(f"ERROR: Gagal memuat mapping kelas dari {Config.CLASS_INDICES_PATH}. Error: {e}")
            _idx_to_class = {} # Set ke dictionary kosong jika gagal

# --- Fungsi Getter untuk Model dan Mapping ---
# Fungsi-fungsi ini memungkinkan modul lain mengakses model dan mapping yang sudah dimuat.
def get_model():
    """Mengembalikan instance model TensorFlow/Keras yang sudah dimuat."""
    # Jika model belum dimuat (misal karena error awal), coba muat lagi
    if _model is None:
        load_model_and_class_indices()
    return _model

def get_idx_to_class_mapping():
    """Mengembalikan mapping kelas (indeks ke nama ras) yang sudah dimuat."""
    # Jika mapping belum dimuat, coba muat lagi
    if _idx_to_class is None:
        load_model_and_class_indices()
    return _idx_to_class

# --- Fungsi Pra-pemrosesan Gambar ---
def preprocess_image(image_path):
    """
    Pra-pemrosesan gambar agar sesuai dengan format input yang diharapkan oleh model CNN.
    Langkah-langkah ini harus identik dengan pra-pemrosesan yang dilakukan saat pelatihan.
    """
    # Membuka gambar menggunakan Pillow, memastikan format RGB (3 channel)
    img = Image.open(image_path).convert('RGB')
    # Mengubah ukuran gambar sesuai dengan dimensi input model (misal 224x224)
    img = img.resize((Config.IMG_WIDTH, Config.IMG_HEIGHT))
    # Mengonversi gambar Pillow menjadi array NumPy
    img_array = image.img_to_array(img)
    # Menambahkan dimensi batch di awal array (misal dari (224,224,3) menjadi (1, 224,224,3))
    # Ini karena model Keras/TensorFlow mengharapkan input dalam bentuk batch, meskipun hanya satu gambar.
    img_array = np.expand_dims(img_array, axis=0)
    # Normalisasi nilai piksel dari rentang [0, 255] ke [0, 1]
    # Langkah ini SANGAT PENTING dan HARUS SAMA PERSIS dengan normalisasi saat pelatihan model!
    img_array = img_array / 255.0
    return img_array

# --- Inisialisasi Otomatis Saat Modul Diimpor ---
# Bagian ini memastikan bahwa folder-folder yang dibutuhkan ada
# dan model dimuat saat modul `model_loader.py` pertama kali diimpor.

# Buat folder sementara untuk unggahan jika belum ada
if not os.path.exists(Config.TEMP_UPLOAD_FOLDER):
    os.makedirs(Config.TEMP_UPLOAD_FOLDER)

# Buat folder permanen untuk gambar history jika belum ada
if not os.path.exists(Config.HISTORY_IMAGES_FOLDER):
    os.makedirs(Config.HISTORY_IMAGES_FOLDER)

# Buat folder untuk file CSV history jika belum ada
# (Meskipun history disimpan di DB, folder ini mungkin masih digunakan untuk CSV backup/debug)
if not os.path.exists(Config.HISTORY_FOLDER):
    os.makedirs(Config.HISTORY_FOLDER)

# Muat model dan mapping kelas saat modul ini diimpor
# Ini akan dijalankan sekali saat aplikasi Flask dimulai.
load_model_and_class_indices()