Dog Breed Classifier Backend
Aplikasi backend berbasis Flask untuk mengklasifikasikan ras anjing dari gambar menggunakan model Convolutional Neural Network (CNN) yang telah dilatih. Aplikasi ini terintegrasi dengan database MySQL untuk menyimpan riwayat prediksi dan menyediakan API endpoint untuk klasifikasi gambar dan pengambilan data historis.

Daftar Isi
Fitur

Struktur Proyek

Teknologi yang Digunakan

Pengaturan dan Instalasi

Prasyarat

Pengaturan Database

Variabel Lingkungan

Menginstal Dependensi

Menjalankan Migrasi

Mempersiapkan Model AI dan Pemetaan Kelas

Menjalankan Aplikasi

API Endpoints

Cara Kerja Program (Penjelasan Kode Rinci)

Pelatihan Model AI (dog-breed-classifier.ipynb)

Inisialisasi Aplikasi (app.py)

Konfigurasi Aplikasi (config.py)

Ekstensi Flask (extensions.py)

Model Database (models_db/prediction_history.py)

Pemuatan Model AI (utils/model_loader.py)

Layanan Klasifikasi (services/classification_service.py)

Rute API (routes/api_routes.py)

Rute Utama (routes/main_routes.py)

Titik Masuk Flask CLI (run.py)

Kontribusi

Lisensi

Fitur
Klasifikasi Ras Anjing: Memprediksi ras anjing dari gambar yang diunggah menggunakan model CNN pra-terlatih (ResNet50V2).

Riwayat Prediksi: Menyimpan hasil klasifikasi (gambar, ras yang diprediksi, tingkat kepercayaan, dan ID pengguna) dalam database MySQL.

Otentikasi Pengguna (JWT): Mengintegrasikan Flask-JWT-Extended untuk otentikasi pengguna opsional pada endpoint prediksi. Pengguna tamu juga dapat melakukan prediksi.

CORS Enabled: Dikonfigurasi untuk menangani Cross-Origin Resource Sharing untuk integrasi frontend yang mulus.

Migrasi Database: Menggunakan Flask-Migrate untuk pengelolaan skema database yang mudah.

Struktur Modular: Diatur ke dalam Blueprints, layanan, dan modul utilitas untuk pemeliharaan yang lebih baik.

Penyimpanan Gambar: Sementara menyimpan gambar yang diunggah untuk diproses dan secara permanen menyimpannya (atau salinannya) untuk melihat riwayat.

Struktur Proyek
my_dog_breed_classifier/
├── app.py                      # Pabrik aplikasi Flask utama
├── config.py                   # Konfigurasi aplikasi
├── extensions.py               # Inisialisasi ekstensi Flask
├── run.py                      # Titik masuk untuk perintah Flask CLI
├── dog-breed-classifier.ipynb  # Jupyter Notebook untuk pelatihan model AI
├── models/                     # Direktori untuk model AI dan metadata
│   ├── dog_breed_classifier_final_model.keras  # Model Keras yang telah dilatih (akan diunduh)
│   └── class_indices.json      # Pemetaan indeks kelas ke nama ras (akan diunduh)
├── models_db/                  # Model database
│   ├── __init__.py
│   ├── user.py                 # Model database pengguna (tidak disediakan, tetapi diharapkan untuk JWT)
│   └── prediction_history.py   # Model database riwayat prediksi
├── routes/                     # Blueprint Flask untuk API dan rute utama
│   ├── __init__.py
│   ├── api_routes.py           # Endpoint API (predict, history)
│   └── main_routes.py          # Rute aplikasi utama (misalnya, halaman indeks)
├── services/                   # Layanan logika bisnis
│   ├── __init__.py
│   └── classification_service.py # Layanan untuk menyimpan/mengambil riwayat prediksi
├── static/                     # File statis (gambar, CSS, JS)
│   ├── history_images/         # Menyimpan gambar yang diklasifikasikan secara permanen
│   └── temp_uploads/           # Menyimpan gambar yang diunggah sementara
├── templates/                  # Templat HTML
│   └── index.html
├── data/                       # Data lain-lain (misalnya, riwayat CSV, jika digunakan)
└── migrations/                 # Skrip migrasi yang dihasilkan Flask-Migrate

Teknologi yang Digunakan
Flask

SQLAlchemy

Flask-Migrate

Flask-CORS

Flask-JWT-Extended

TensorFlow/Keras

Numpy

Pillow (PIL)

MySQL (via PyMySQL)

Werkzeug

Pandas

KaggleHub

Scikit-learn

Matplotlib

Python-dotenv

Pengaturan dan Instalasi
Prasyarat
Python 3.8+: Pastikan Anda memiliki versi Python yang kompatibel terinstal.

MySQL Server: Anda memerlukan server database MySQL yang berjalan. Alat seperti Laragon (Windows), MAMP (macOS), atau Docker dapat digunakan untuk pengaturan yang mudah.

Pengaturan Database
Buat Database MySQL:
Menggunakan klien MySQL Anda (misalnya, phpMyAdmin, MySQL Workbench, atau command line), buat database baru.
Misalnya, jika Anda menggunakan Laragon, Anda dapat membuat database bernama dog_classifier_db.

CREATE DATABASE dog_classifier_db;

Pengguna Database (Opsional tetapi Direkomendasikan):
Jika Anda tidak menggunakan pengguna root default tanpa kata sandi (umum dalam pengembangan lokal dengan Laragon), buat pengguna khusus dan berikan hak istimewa ke database Anda.

CREATE USER 'your_db_user'@'localhost' IDENTIFIED BY 'your_db_password';
GRANT ALL PRIVILEGES ON dog_classifier_db.* TO 'your_db_user'@'localhost';
FLUSH PRIVILEGES;

Perbarui root dan your_db_password di config.py sesuai.

Variabel Lingkungan
Sangat disarankan untuk menggunakan variabel lingkungan untuk informasi sensitif seperti SECRET_KEY, JWT_SECRET_KEY, dan DATABASE_URL. Buat file .env di direktori root proyek Anda (misalnya, my_dog_breed_classifier/.env):

SECRET_KEY='kunci_rahasia_sangat_rahasia_anda_di_sini'
JWT_SECRET_KEY='kunci_rahasia_jwt_sangat_rahasia_lainnya'
DATABASE_URL='mysql+pymysql://root:@localhost:3306/dog_classifier_db'
# Jika Anda membuat pengguna: mysql+pymysql://your_db_user:your_db_password@localhost:3306/dog_classifier_db

Catatan: File config.py memiliki nilai default untuk ini, tetapi menggunakan variabel lingkungan adalah praktik yang lebih aman, terutama dalam produksi. Jika DATABASE_URL tidak diatur dalam .env, itu akan kembali ke mysql+pymysql://root:@localhost:3306/dog_classifier_db.

Menginstal Dependensi
Arahkan ke direktori root proyek Anda di terminal dan instal paket Python yang diperlukan. Pastikan Anda memiliki file requirements.txt yang berisi semua dependensi yang disebutkan di bagian Teknologi yang Digunakan.

pip install -r requirements.txt

Menjalankan Migrasi
Setelah mengatur database dan menginstal dependensi, inisialisasi Flask-Migrate dan buat tabel database.

Inisialisasi Migrasi:

flask db init

Ini membuat folder migrations.

Buat Migrasi Awal:

flask db migrate -m "Initial migration"

Ini menghasilkan skrip migrasi berdasarkan models_db Anda.

Terapkan Migrasi:

flask db upgrade

Ini akan membuat tabel prediction_history (dan tabel users jika user.py didefinisikan dengan benar) di database MySQL Anda.

Mempersiapkan Model AI dan Pemetaan Kelas
Model AI (dog_breed_classifier_final_model.keras) dan pemetaan kelasnya (class_indices.json) dihasilkan dengan menjalankan Jupyter Notebook dog-breed-classifier.ipynb.

Jalankan Jupyter Notebook:
Buka dog-breed-classifier.ipynb di lingkungan Jupyter (misalnya, Jupyter Lab, VS Code dengan ekstensi Jupyter, atau Google Colab/Kaggle Notebooks).
Jalankan semua sel. Notebook akan mengunduh dataset Stanford Dogs, memprosesnya, melatih model ResNet50V2, mengevaluasinya, dan akhirnya menyimpan model dan file pemetaan kelas ke direktori keluaran.

Unduh File Model:
Setelah notebook selesai, unduh dog_breed_classifier_final_model.keras dan class_indices.json dari keluaran notebook.

Tempatkan File Model:
Tempatkan kedua file ini ke dalam direktori models/ di dalam proyek backend Anda:

my_dog_breed_classifier/
└── models/
    ├── dog_breed_classifier_final_model.keras
    └── class_indices.json

Menjalankan Aplikasi
Ada dua cara utama untuk menjalankan aplikasi Flask:

Menggunakan flask run (Direkomendasikan untuk Pengembangan):
Setel variabel lingkungan FLASK_APP dan kemudian jalankan:

export FLASK_APP=run.py # Untuk Linux/macOS
# set FLASK_APP=run.py # Untuk Windows CMD
# $env:FLASK_APP="run.py" # For Windows PowerShell

flask run

Ini akan memulai server pengembangan Flask, biasanya di http://127.0.0.1:5000/.

Menjalankan Langsung app.py:

python app.py

Ini juga memulai server pengembangan. File app.py dikonfigurasi untuk berjalan langsung saat dieksekusi sebagai skrip utama.

Anda akan melihat output yang menunjukkan bahwa model AI dan pemetaan kelas telah berhasil dimuat:

Model AI berhasil dimuat dari: /path/to/my_dog_breed_classifier/models/dog_breed_classifier_final_model.keras
Model dimuat untuk inferensi GPU. (or CPU)
Mapping kelas berhasil dimuat dari: /path/to/my_dog_breed_classifier/models/class_indices.json

API Endpoints
Semua API endpoint diawali dengan /api.

GET /

Deskripsi: Titik masuk utama untuk aplikasi web. Mengembalikan index.html.

File: routes/main_routes.py

Akses: Publik.

Respons: Merender templates/index.html.

POST /api/predict

Deskripsi: Mengklasifikasikan gambar anjing yang diunggah dan menyimpan riwayat prediksi.

File: routes/api_routes.py

Akses: Publik (otentikasi opsional). Jika token JWT yang valid disediakan, user_id disimpan; jika tidak, nilainya adalah None.

Permintaan: multipart/form-data dengan bidang file yang berisi gambar.

Respons:

200 OK: {"predicted_breed": "NamaRas", "confidence": "XX.YY%", "image_url": "/static/history_images/nama_file_unik.jpg"}

400 Bad Request: Jika tidak ada file yang diunggah atau nama file kosong.

500 Internal Server Error: Jika model AI gagal dimuat atau terjadi kesalahan pemrosesan lainnya.

GET /api/history

Deskripsi: Mengambil semua riwayat prediksi ras anjing yang tersimpan.

File: routes/api_routes.py

Akses: Publik (tidak memerlukan otentikasi).

Permintaan: Tidak ada.

Respons:

200 OK: [{"id": 1, "timestamp": "YYYY-MM-DD HH:MM:SS", "image_filename": "nama_file.jpg", "image_url": "/static/history_images/nama_file.jpg", "predicted_breed": "NamaRas", "confidence": "XX.YY%", "user_id": 123}] (daftar objek riwayat).

500 Internal Server Error: Jika pengambilan database gagal.

Cara Kerja Program (Penjelasan Kode Rinci)
Bagian ini merinci alur dan tujuan dari setiap file penting, termasuk cuplikan kode yang relevan dan penjelasannya.

Pelatihan Model AI (dog-breed-classifier.ipynb)
Notebook Jupyter ini adalah tempat model AI dilatih dari awal menggunakan dataset Stanford Dogs.

1. Import Library dan Cek GPU
Bagian ini mengimpor semua pustaka Python yang diperlukan untuk pemrosesan data, membangun model, pelatihan, dan evaluasi. Ini juga memeriksa apakah TensorFlow dapat mendeteksi dan menggunakan GPU, yang sangat penting untuk mempercepat pelatihan model deep learning.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.applications import ResNet50V2

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import kagglehub

print("TensorFlow version:", tf.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

numpy, pandas, matplotlib.pyplot, os, json: Pustaka standar untuk manipulasi data, visualisasi, operasi sistem file, dan JSON.

tensorflow, keras.*: Pustaka utama untuk membangun dan melatih model deep learning.

ImageDataGenerator: Digunakan untuk pra-pemrosesan gambar dan augmentasi data.

Sequential, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization: Lapisan-lapisan yang digunakan untuk membangun arsitektur CNN.

Adam: Optimizer untuk pelatihan model.

EarlyStopping, ReduceLROnPlateau, ModelCheckpoint: Callbacks untuk mengontrol proses pelatihan.

ResNet50V2: Model CNN pra-terlatih yang akan digunakan untuk transfer learning.

train_test_split, classification_report: Dari sklearn untuk membagi data dan mengevaluasi model.

kagglehub: Untuk mengunduh dataset dari Kaggle.

2. Unduh Dataset
Bagian ini menggunakan kagglehub untuk mengunduh dataset Stanford Dogs langsung ke lingkungan notebook. Ini memastikan bahwa data tersedia untuk pemrosesan lebih lanjut.

print("\nMengunduh dataset Stanford Dogs dari Kaggle...")
path_ke_dataset_root = kagglehub.dataset_download("jessicali9530/stanford-dogs-dataset")

print(f"Dataset berhasil diunduh ke: {path_ke_dataset_root}")

path_ke_gambar = os.path.join(path_ke_dataset_root, 'images', 'Images')

if not os.path.exists(path_ke_gambar):
    print(f"Error: Folder gambar tidak ditemukan di '{path_ke_gambar}'. Silakan periksa struktur dataset yang diunduh.")
else:
    print(f"Folder gambar ditemukan di: {path_ke_gambar}")

kagglehub.dataset_download(...): Fungsi untuk mengunduh dataset dari Kaggle.

os.path.join(...): Digunakan untuk membangun path file yang cross-platform.

Pengecekan os.path.exists(): Memastikan folder gambar yang diharapkan benar-benar ada.

3. Pra-pemrosesan Data dan Augmentasi
Bagian ini mengumpulkan path gambar dan label, membuat DataFrame untuk manajemen data yang mudah, dan kemudian mengonfigurasi ImageDataGenerator untuk pra-pemrosesan dan augmentasi data. Augmentasi data sangat penting untuk mencegah overfitting dengan menciptakan variasi gambar baru.

print("Memulai pra-pemrosesan data gambar...")

image_paths = []
labels = []

for breed_folder in os.listdir(path_ke_gambar):
    breed_path = os.path.join(path_ke_gambar, breed_folder)
    if os.path.isdir(breed_path):
        breed_name = breed_folder.split('-')[1]
        for img_name in os.listdir(breed_path):
            image_paths.append(os.path.join(breed_path, img_name))
            labels.append(breed_name)

df = pd.DataFrame({'path': image_paths, 'label': labels})
print(f"Total gambar yang ditemukan: {len(df)}")
print(f"Jumlah ras anjing unik: {df['label'].nunique()}")

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = df['label'].nunique()

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

validation_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='path',
    y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='path',
    y_col='label',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

class_indices = train_generator.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}
print("\nMapping kelas (indeks angka ke nama ras):")
print(idx_to_class)

os.listdir(), os.path.isdir(): Untuk menjelajahi struktur folder dataset.

pd.DataFrame(): Membuat DataFrame Pandas dari path gambar dan label.

train_test_split(): Membagi DataFrame menjadi set pelatihan dan pengujian. stratify=df['label'] memastikan distribusi kelas yang sama di kedua set.

ImageDataGenerator():

rescale=1./255: Normalisasi nilai piksel dari [0, 255] menjadi [0, 1].

rotation_range, width_shift_range, height_shift_range, shear_range, zoom_range, horizontal_flip: Parameter untuk augmentasi data.

validation_split: Memisahkan sebagian data pelatihan untuk validasi selama pelatihan.

flow_from_dataframe(): Metode generator yang mengambil data dari DataFrame dan menghasilkan batch gambar yang telah diproses.

idx_to_class: Membuat pemetaan dari indeks numerik ke nama ras, yang penting untuk menafsirkan output model.

4. Membangun Model CNN dengan Transfer Learning
Bagian ini membangun arsitektur model menggunakan pendekatan transfer learning dengan ResNet50V2.

print("Membangun model CNN dengan Transfer Learning...")

base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

ResNet50V2(weights='imagenet', include_top=False, input_shape=...): Menginisialisasi model ResNet50V2 yang telah dilatih pada dataset ImageNet. include_top=False berarti lapisan klasifikasi asli model tidak disertakan, memungkinkan kita menambahkan lapisan khusus kita sendiri.

for layer in base_model.layers: layer.trainable = False: Membekukan bobot model dasar. Ini mencegah bobot yang sudah dilatih sebelumnya berubah selama pelatihan awal, menjaga fitur-fitur yang telah dipelajari dengan baik.

Sequential([...]): Membangun model sekuensial dengan menambahkan lapisan-lapisan di atas base_model:

Flatten(): Mengubah output 2D/3D dari base model menjadi 1D.

Dense(512, activation='relu'): Lapisan terhubung penuh dengan 512 neuron dan fungsi aktivasi ReLU.

BatchNormalization(): Normalisasi batch untuk menstabilkan dan mempercepat pelatihan.

Dropout(0.5): Lapisan dropout (50%) untuk mencegah overfitting dengan secara acak menonaktifkan neuron.

Dense(NUM_CLASSES, activation='softmax'): Lapisan output dengan jumlah neuron yang sama dengan jumlah kelas anjing, menggunakan aktivasi softmax untuk menghasilkan probabilitas kelas.

model.compile(...): Mengonfigurasi model untuk pelatihan.

optimizer=Adam(learning_rate=0.0001): Menggunakan optimizer Adam dengan learning rate kecil, yang umum untuk transfer learning.

loss='categorical_crossentropy': Fungsi loss yang sesuai untuk masalah klasifikasi multi-kelas dengan label one-hot encoded.

metrics=['accuracy']: Metrik yang akan dipantau selama pelatihan.

model.summary(): Menampilkan ringkasan arsitektur model, termasuk jumlah parameter yang dapat dilatih dan tidak dapat dilatih.

5. Menyiapkan Callbacks Pelatihan
Callbacks adalah fungsi yang dipanggil pada berbagai tahap pelatihan (misalnya, di akhir setiap epoch) untuk melakukan tindakan tertentu.

print("--- Menyiapkan callbacks pelatihan ---")

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

checkpoint_path = "best_dog_breed_classifier.keras"
model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

callbacks = [early_stopping, reduce_lr, model_checkpoint]

EarlyStopping: Menghentikan pelatihan lebih awal jika loss validasi tidak membaik selama sejumlah epoch (patience). restore_best_weights=True akan mengembalikan bobot model dari epoch terbaik.

ReduceLROnPlateau: Mengurangi learning rate jika metrik yang dipantau (val_accuracy) berhenti membaik.

ModelCheckpoint: Menyimpan bobot model terbaik (berdasarkan val_accuracy) ke file tertentu.

6. Melatih Model
Bagian ini memulai proses pelatihan model menggunakan data yang dihasilkan oleh generator dan callbacks yang telah disiapkan.

print("--- Memulai pelatihan model ---")

EPOCHS = 50

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

print("Pelatihan model selesai.")

model.fit(): Fungsi untuk melatih model.

train_generator: Sumber data pelatihan.

epochs: Jumlah epoch pelatihan.

validation_data: Data yang digunakan untuk memantau kinerja model pada data yang tidak terlihat selama pelatihan.

callbacks: Daftar callbacks yang akan digunakan.

verbose=1: Menampilkan progress pelatihan di setiap epoch.

history: Objek yang mengembalikan riwayat pelatihan, termasuk loss dan accuracy untuk pelatihan dan validasi di setiap epoch.

7. Evaluasi Model dan Visualisasi Hasil
Setelah pelatihan, model dievaluasi pada data pengujian yang sepenuhnya tidak terlihat untuk mengukur kinerja sebenarnya. Hasil pelatihan juga divisualisasikan.

print("--- Memulai evaluasi model ---")

try:
    model.load_weights(checkpoint_path)
    print(f"Berhasil memuat bobot model terbaik dari: {checkpoint_path}")
except Exception as e:
    print(f"Gagal memuat bobot terbaik. Menggunakan model yang terakhir dilatih. Error: {e}")

loss, accuracy = model.evaluate(test_generator)
print(f"\nHasil Evaluasi pada Data Testing:")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Akurasi Pelatihan')
plt.plot(history.history['val_accuracy'], label='Akurasi Validasi')
plt.title('Grafik Akurasi Model')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss Pelatihan')
plt.plot(history.history['val_loss'], label='Loss Validasi')
plt.title('Grafik Loss Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

print("\n--- Membuat prediksi pada data testing untuk Laporan Klasifikasi ---")
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

true_labels_names = [idx_to_class[i] for i in y_true]
predicted_labels_names = [idx_to_class[i] for i in y_pred]

print("\nLaporan Klasifikasi Detail:")
print(classification_report(true_labels_names, predicted_labels_names, target_names=list(idx_to_class.values())))

model.load_weights(checkpoint_path): Memuat bobot model terbaik yang disimpan oleh ModelCheckpoint selama pelatihan.

model.evaluate(test_generator): Mengevaluasi kinerja model pada data pengujian.

matplotlib.pyplot: Digunakan untuk memplot grafik akurasi dan loss pelatihan/validasi, membantu memvisualisasikan progress pelatihan.

model.predict(test_generator): Membuat prediksi pada data pengujian.

np.argmax(y_pred_probs, axis=1): Mengambil indeks kelas dengan probabilitas tertinggi dari prediksi.

classification_report(): Menghasilkan laporan klasifikasi rinci yang mencakup precision, recall, f1-score, dan support untuk setiap kelas.

8. Menyimpan Model dan Metadata
Langkah terakhir adalah menyimpan model yang telah dilatih dan pemetaan kelas ke file, sehingga dapat dimuat dan digunakan oleh aplikasi backend.

print("--- Menyimpan model dan metadata ---")

model_save_path = 'dog_breed_classifier_final_model.keras'
model.save(model_save_path)
print(f"Model akhir berhasil disimpan ke: {model_save_path}")

with open('class_indices.json', 'w') as f:
    json.dump(idx_to_class, f)
print(f"Mapping kelas berhasil disimpan ke: class_indices.json")

print("\nProses di Kaggle Notebook selesai.")
print("File yang perlu Anda unduh dari bagian 'Output' pada Kaggle Notebook adalah:")
print(f"- {model_save_path}")
print(f"- class_indices.json")

model.save(model_save_path): Menyimpan seluruh model (arsitektur, bobot, konfigurasi optimizer) ke file .keras.

json.dump(idx_to_class, f): Menyimpan kamus idx_to_class (pemetaan indeks ke nama ras) ke file JSON. File ini penting agar backend dapat menerjemahkan output numerik model menjadi nama ras yang dapat dibaca manusia.

Inisialisasi Aplikasi (app.py)
File app.py berisi fungsi create_app() yang berfungsi sebagai application factory untuk Flask. Ini adalah praktik terbaik untuk aplikasi Flask yang lebih besar dan modular.

# my_dog_breed_classifier/app.py

from flask import Flask, jsonify
import os
from flask_cors import CORS
from config import Config
from routes import register_blueprints
from extensions import db, jwt, cors, migrate

def create_app():
    """
    Fungsi pabrik untuk membuat dan mengonfigurasi instance aplikasi Flask.
    Ini adalah praktik terbaik untuk aplikasi Flask yang lebih besar dan modular.
    """
    app = Flask(__name__, static_folder='static', instance_relative_config=True)
    app.config.from_object(Config)

    # Inisialisasi Ekstensi dengan Aplikasi Flask
    db.init_app(app)
    jwt.init_app(app)
    cors.init_app(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"], supports_credentials=True, allow_headers=["Authorization", "Content-Type"])
    migrate.init_app(app, db)

    # JWT CALLBACKS GLOBAL
    @jwt.unauthorized_loader
    def unauthorized_response(callback):
        """Callback ketika token JWT hilang atau tidak valid."""
        return jsonify({"message": "Token akses hilang atau tidak valid. Mohon login."}), 401

    @jwt.invalid_token_loader
    def invalid_token_response(callback):
        """Callback ketika token JWT formatnya salah atau rusak/kadaluarsa."""
        return jsonify({"message": "Token akses tidak valid atau telah kadaluarsa. Mohon login kembali."}), 401

    @jwt.needs_fresh_token_loader
    def needs_fresh_token_response(callback):
        """Callback ketika token yang diberikan tidak 'fresh' tapi rute membutuhkannya."""
        return jsonify({"message": "Token segar diperlukan untuk mengakses rute ini."}), 401

    @jwt.revoked_token_loader
    def revoked_token_response(jwt_header, jwt_payload):
        """Callback ketika token yang diberikan telah dicabut."""
        return jsonify({"message": "Token akses telah dicabut."}), 401

    # Pastikan Folder-folder yang Dibutuhkan Ada
    if not os.path.exists(app.config['TEMP_UPLOAD_FOLDER']):
        os.makedirs(app.config['TEMP_UPLOAD_FOLDER'])
    if not os.path.exists(app.config['HISTORY_IMAGES_FOLDER']):
        os.makedirs(app.config['HISTORY_IMAGES_FOLDER'])
    if not os.path.exists(app.config['HISTORY_FOLDER']):
        os.makedirs(app.config['HISTORY_FOLDER'])

    # Membuat Tabel Database
    with app.app_context():
        from models_db.user import User
        from models_db.prediction_history import PredictionHistory
        db.create_all()

    # Daftarkan Semua Blueprints
    register_blueprints(app)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=app.config['DEBUG'], host=app.config['HOST'], port=app.config['PORT'])

create_app(): Fungsi ini adalah "pabrik" yang membuat dan mengonfigurasi instansi aplikasi Flask. Ini memungkinkan aplikasi menjadi modular dan mudah diuji.

app.config.from_object(Config): Memuat semua pengaturan dari kelas Config yang didefinisikan di config.py.

db.init_app(app), jwt.init_app(app), cors.init_app(app, ...), migrate.init_app(app, db): Baris-baris ini menginisialisasi ekstensi Flask (SQLAlchemy, JWTManager, CORS, Flask-Migrate) dengan instansi aplikasi Flask. Ini mengintegrasikan fungsionalitas ekstensi ke dalam aplikasi.

@jwt.unauthorized_loader dan callbacks JWT lainnya: Ini adalah decorator yang mendaftarkan fungsi callback untuk menangani berbagai skenario error terkait JWT (misalnya, token hilang, tidak valid, kadaluarsa). Mereka memastikan respons JSON yang konsisten untuk error otentikasi.

os.makedirs(...): Baris-baris ini memastikan bahwa direktori yang diperlukan untuk menyimpan file sementara dan gambar riwayat ada. Jika tidak ada, mereka akan dibuat secara otomatis.

with app.app_context(): db.create_all(): Bagian ini sangat penting untuk inisialisasi database. db.create_all() akan membuat semua tabel database yang didefinisikan oleh model SQLAlchemy (seperti User dan PredictionHistory) jika tabel tersebut belum ada. Ini harus dipanggil di dalam application context karena berinteraksi dengan database yang terkait dengan aplikasi.

register_blueprints(app): Fungsi ini (dari routes/__init__.py) mendaftarkan semua Blueprints rute (misalnya, api_bp, main_bp) ke aplikasi Flask, mengorganisir rute ke dalam modul yang logis.

if __name__ == '__main__': app.run(...): Blok ini memastikan bahwa aplikasi hanya berjalan ketika app.py dieksekusi langsung (misalnya, python app.py), bukan ketika diimpor sebagai modul. Ini memulai server pengembangan Flask.

Konfigurasi Aplikasi (config.py)
File config.py mendefinisikan kelas Config yang menampung semua pengaturan global dan path penting untuk aplikasi.

import os
from datetime import timedelta

class Config:
    """
    Kelas konfigurasi untuk aplikasi Flask.
    Menyimpan semua path penting dan pengaturan global.
    """
    # Mendapatkan path absolut ke direktori dasar proyek (tempat file config.py berada)
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    SECRET_KEY = os.environ.get('SECRET_KEY', '1a2b3c4d5e6f7g8h9i0j11213141516171819202122232425')
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000

    # Konfigurasi Database (MySQL via PyMySQL)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'mysql+pymysql://root:@localhost:3306/dog_classifier_db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'a9b8c7d6e5f4g3h2i1j0192837465abcd1234567890abcdef')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=3)

    # Path untuk Model AI
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'dog_breed_classifier_final_model.keras')
    CLASS_INDICES_PATH = os.path.join(BASE_DIR, 'models', 'class_indices.json')

    # UKURAN GAMBAR
    IMG_HEIGHT = 224
    IMG_WIDTH = 224

    # Path untuk Folder Unggahan Gambar dan History Gambar
    TEMP_UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'temp_uploads')
    HISTORY_IMAGES_FOLDER = os.path.join(BASE_DIR, 'static', 'history_images')

    # Path untuk History Klasifikasi (jika masih menggunakan CSV paralel)
    HISTORY_FOLDER = os.path.join(BASE_DIR, 'data')
    HISTORY_FILE = os.path.join(HISTORY_FOLDER, 'classification_history.csv')

BASE_DIR: Menentukan direktori dasar proyek, yang digunakan sebagai referensi untuk semua path relatif lainnya.

SECRET_KEY & JWT_SECRET_KEY: Kunci rahasia yang digunakan untuk keamanan sesi Flask dan penandatanganan token JWT. Penting untuk menjadikannya kuat dan tidak mudah ditebak. Nilai ini diambil dari variabel lingkungan untuk produksi, dengan fallback untuk pengembangan.

SQLALCHEMY_DATABASE_URI: String koneksi ke database MySQL. Ini menentukan jenis database (mysql+pymysql), kredensial (root:@localhost:3306), dan nama database (dog_classifier_db).

MODEL_PATH & CLASS_INDICES_PATH: Path absolut ke file model AI (.keras) dan file JSON yang berisi pemetaan indeks kelas ke nama ras. Ini memastikan aplikasi dapat menemukan dan memuat model yang dilatih.

IMG_HEIGHT & IMG_WIDTH: Dimensi gambar yang diharapkan oleh model AI. Semua gambar yang diunggah akan diubah ukurannya ke dimensi ini sebelum diproses.

TEMP_UPLOAD_FOLDER & HISTORY_IMAGES_FOLDER: Path ke folder tempat gambar yang diunggah sementara disimpan sebelum diproses, dan tempat gambar yang diklasifikasikan disimpan secara permanen untuk riwayat.

Ekstensi Flask (extensions.py)
File ini menginisialisasi instansi ekstensi Flask yang akan digunakan di seluruh aplikasi.

# extensions.py
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from flask_jwt_extended import JWTManager

db = SQLAlchemy()
migrate = Migrate()
cors = CORS()
jwt = JWTManager()

def init_app(app):
    """
    Menginisialisasi semua ekstensi Flask dengan instance aplikasi yang diberikan.
    """
    db.init_app(app)
    migrate.init_app(app, db)
    cors.init_app(app)
    jwt.init_app(app)

db = SQLAlchemy(): Membuat instansi SQLAlchemy yang akan dihubungkan ke aplikasi Flask. Ini adalah ORM yang memungkinkan Anda berinteraksi dengan database menggunakan objek Python alih-alih SQL mentah.

migrate = Migrate(): Membuat instansi Flask-Migrate yang digunakan untuk mengelola migrasi skema database. Ini memungkinkan Anda membuat perubahan pada model database Anda dan menerapkan perubahan tersebut ke database secara terstruktur.

cors = CORS(): Membuat instansi Flask-CORS yang memungkinkan aplikasi Flask Anda merespons permintaan cross-origin (misalnya, dari frontend yang berjalan di domain berbeda).

jwt = JWTManager(): Membuat instansi Flask-JWT-Extended yang menyediakan dukungan untuk token Web JSON (JWT) untuk otentikasi pengguna.

init_app(app): Fungsi ini adalah pola umum dalam aplikasi Flask yang lebih besar. Ini memungkinkan ekstensi diinisialisasi secara terpisah dari objek aplikasi utama, mempromosikan modularitas dan pengujian. app dilewatkan ke fungsi ini, dan kemudian setiap ekstensi diinisialisasi dengan aplikasi tersebut.

Model Database (models_db/prediction_history.py)
File ini mendefinisikan model SQLAlchemy untuk tabel prediction_history di database.

# my_dog_breed_classifier/models_db/prediction_history.py

# Import db secara absolut dari extensions.py
from extensions import db
from datetime import datetime # Untuk default timestamp

class PredictionHistory(db.Model):
    __tablename__ = 'prediction_history' # Nama tabel di database

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True) # Foreign Key ke tabel users
    timestamp = db.Column(db.DateTime, nullable=False, default=db.func.now()) # Waktu prediksi
    image_filename = db.Column(db.String(255), nullable=False) # Nama file gambar di folder static/history_images
    image_url = db.Column(db.String(255), nullable=False) # URL relatif gambar untuk frontend
    predicted_breed = db.Column(db.String(80), nullable=False)
    confidence = db.Column(db.Float, nullable=False) # Kepercayaan prediksi (sebagai float)

    def __repr__(self):
        return f'<Prediction {self.predicted_breed} by User {self.user_id}>'

class PredictionHistory(db.Model):: Mendefinisikan kelas model yang mewarisi dari db.Model SQLAlchemy, yang memetakannya ke tabel database.

__tablename__ = 'prediction_history': Menentukan nama tabel di database.

id = db.Column(db.Integer, primary_key=True): Mendefinisikan kolom id sebagai bilangan bulat dan kunci utama, yang secara otomatis akan bertambah.

user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True): Kolom user_id adalah kunci asing yang menautkan ke kolom id di tabel users. nullable=True berarti prediksi dapat dibuat oleh pengguna yang tidak diautentikasi (tamu).

timestamp = db.Column(db.DateTime, nullable=False, default=db.func.now()): Kolom ini menyimpan waktu prediksi. db.func.now() secara otomatis mengisi waktu saat ini saat entri baru dibuat.

image_filename = db.Column(db.String(255), nullable=False): Menyimpan nama file gambar asli.

image_url = db.Column(db.String(255), nullable=False): Menyimpan URL relatif gambar yang disimpan di folder statis, yang dapat diakses oleh frontend.

predicted_breed = db.Column(db.String(80), nullable=False): Menyimpan nama ras anjing yang diprediksi.

confidence = db.Column(db.Float, nullable=False): Menyimpan tingkat kepercayaan prediksi sebagai angka pecahan.

__repr__(self): Metode ini menyediakan representasi string yang mudah dibaca dari objek PredictionHistory, berguna untuk debugging.

Pemuatan Model AI (utils/model_loader.py)
Modul ini bertanggung jawab untuk memuat model AI dan pemetaan kelas ke memori dan menyediakan fungsi pra-pemrosesan gambar.

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os
from PIL import Image

# Import kelas Config dari file config.py (Import absolut yang benar)
from config import Config

# Variabel Global untuk Model dan Mapping Kelas
# Ini akan menyimpan instance model TensorFlow/Keras dan mapping kelas
# setelah dimuat pertama kali. Tujuannya agar tidak perlu memuat ulang
# setiap kali ada permintaan prediksi, yang akan sangat lambat.
_model = None
_idx_to_class = None

# Fungsi untuk Memuat Model dan Mapping Kelas
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

# Fungsi Getter untuk Model dan Mapping
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

# Fungsi Pra-pemrosesan Gambar
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

# Inisialisasi Otomatis Saat Modul Diimpor
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

_model = None, _idx_to_class = None: Variabel global ini digunakan untuk menyimpan model dan pemetaan kelas setelah dimuat pertama kali. Ini mencegah aplikasi memuat ulang model yang besar di setiap permintaan prediksi, yang akan sangat lambat.

load_model_and_class_indices(): Fungsi ini adalah inti dari pemuatan model. Ini mencoba memuat model Keras dari Config.MODEL_PATH dan pemetaan kelas dari Config.CLASS_INDICES_PATH. Ini mencakup penanganan error jika file tidak ditemukan atau rusak.

get_model() & get_idx_to_class_mapping(): Ini adalah fungsi getter yang aman. Setiap kali modul lain membutuhkan model atau pemetaan kelas, mereka memanggil fungsi ini. Fungsi ini akan memuat model jika belum dimuat.

preprocess_image(image_path): Fungsi ini mengambil path ke gambar, membukanya dengan Pillow, memastikan format RGB, mengubah ukurannya ke dimensi yang diharapkan model (IMG_WIDTH, IMG_HEIGHT), mengonversinya menjadi array NumPy, menambahkan dimensi batch (karena model mengharapkan input dalam bentuk batch), dan menormalkan nilai piksel ke rentang [0, 1]. Normalisasi ini sangat penting dan harus sama persis dengan yang digunakan selama pelatihan model.

if not os.path.exists(...): Bagian ini memastikan bahwa folder yang diperlukan untuk unggahan sementara dan penyimpanan gambar riwayat ada. Ini dijalankan secara otomatis saat modul model_loader.py pertama kali diimpor oleh aplikasi.

load_model_and_class_indices() (di bagian bawah file): Baris ini memastikan bahwa model dan pemetaan kelas dimuat segera setelah modul model_loader.py diimpor oleh aplikasi Flask, sehingga model siap digunakan sejak aplikasi dimulai.

Layanan Klasifikasi (services/classification_service.py)
Modul ini berisi logika bisnis untuk menyimpan dan mengambil riwayat prediksi, berinteraksi langsung dengan model database.

# my_dog_breed_classifier/services/classification_service.py

from extensions import db # Mengimpor instance 'db' dari extensions.py
from models_db.prediction_history import PredictionHistory # Mengimpor model PredictionHistory
import pandas as pd
import os
from datetime import datetime

from config import Config # Mengimpor konfigurasi

class ClassificationService:
    """
    Layanan yang menangani logika bisnis terkait klasifikasi dan history prediksi.
    """

    @staticmethod
    def save_prediction_history(user_id, image_filename, image_url, predicted_breed, confidence):
        """
        Menyimpan hasil prediksi ke database.
        Menerima user_id yang bisa berupa None (untuk guest user).
        Mengembalikan dictionary {'success': bool, 'message': str}.
        """
        print(f"DEBUG: Mencoba menyimpan prediksi untuk {predicted_breed} (user_id: {user_id}, file: {image_filename})")
        try:
            # Buat objek PredictionHistory baru
            new_history = PredictionHistory(
                user_id=user_id, # user_id bisa None sekarang
                image_filename=image_filename,
                image_url=image_url,
                predicted_breed=predicted_breed,
                confidence=confidence
            )
            # Tambahkan ke session database dan commit
            db.session.add(new_history)
            print("DEBUG: new_history ditambahkan ke sesi DB. Mencoba commit...")
            db.session.commit()
            print(f"DEBUG: Commit berhasil untuk prediksi {predicted_breed}!")
            return {'success': True, 'message': 'History berhasil disimpan.'}
        except Exception as e:
            db.session.rollback() # Rollback jika terjadi error
            print(f"ERROR: Terjadi kesalahan saat menyimpan history ke DB: {e}")
            return {'success': False, 'message': 'Gagal menyimpan history.'}

    @staticmethod
    def get_user_history(user_id):
        """
        Mengambil history prediksi untuk pengguna tertentu dari database.
        Mengembalikan list of dictionaries.
        """
        # Penting: Jika user_id adalah None, ini seharusnya tidak dipanggil jika Anda ingin filter per user
        # Untuk rute /api/history yang publik, kita menggunakan get_all_history()
        if user_id is None:
            print("Peringatan: get_user_history dipanggil dengan user_id None. Gunakan get_all_history untuk publik.")
            return []

        print(f"DEBUG: Memanggil get_user_history() untuk user_id: {user_id}...")
        try:
            # Query semua record history untuk user_id tertentu, diurutkan berdasarkan timestamp terbaru
            history_records = PredictionHistory.query.filter_by(user_id=user_id).order_by(PredictionHistory.timestamp.desc()).all()
            print(f"DEBUG: Ditemukan {len(history_records)} record history dari DB untuk user {user_id}.")

            history_list = []
            for record in history_records:
                # Format data agar mudah dikonsumsi frontend
                history_list.append({
                    'id': record.id,
                    'timestamp': record.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    'image_filename': record.image_filename,
                    'image_url': f"/{record.image_url}", # Tambahkan '/' di awal untuk URL relatif yang benar di frontend
                    'predicted_breed': record.predicted_breed,
                    'confidence': f"{record.confidence*100:.2f}%" # Format confidence ke string persentase
                })
            print(f"DEBUG: history_list siap untuk user {user_id}: {len(history_list)} item.")
            return history_list
        except Exception as e:
            print(f"ERROR: Terjadi kesalahan saat mengambil history user {user_id} dari DB: {e}")
            return []

    @staticmethod
    def get_all_history():
        """
        Mengambil SEMUA history prediksi dari database (untuk akses publik).
        Mengembalikan list of dictionaries.
        """
        print("DEBUG: Memanggil get_all_history()...")
        try:
            # Query semua record history, diurutkan berdasarkan timestamp terbaru
            history_records = PredictionHistory.query.order_by(PredictionHistory.timestamp.desc()).all()
            print(f"DEBUG: Ditemukan {len(history_records)} record history dari DB (semua).")

            history_list = []
            for record in history_records:
                history_list.append({
                    'id': record.id,
                    'timestamp': record.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    'image_filename': record.image_filename,
                    'image_url': f"/{record.image_url}", # Tambahkan '/' di awal untuk URL relatif yang benar di frontend
                    'predicted_breed': record.predicted_breed,
                    'confidence': f"{record.confidence*100:.2f}%",
                    'user_id': record.user_id # Sertakan user_id untuk info, bisa None
                })
            print(f"DEBUG: history_list (all) siap: {len(history_list)} item.")
            return history_list
        except Exception as e:
            print(f"ERROR: Terjadi kesalahan saat mengambil semua history dari DB: {e}")
            return []

    # Metode untuk History yang Disimpan di CSV (jika masih digunakan paralel)
    @staticmethod
    def save_prediction_history_to_csv(timestamp_str, unique_filename, relative_image_url, predicted_breed, confidence, user_id):
        """
        Menyimpan hasil prediksi ke file CSV (opsional, jika masih digunakan paralel dengan DB).
        """
        history_data = {
            'timestamp': timestamp_str,
            'image_filename': unique_filename,
            'image_url': relative_image_url,
            'predicted_breed': predicted_breed,
            'confidence': confidence,
            'user_id': user_id
        }
        if os.path.exists(Config.HISTORY_FILE) and os.path.getsize(Config.HISTORY_FILE) > 0:
            history_df = pd.DataFrame([history_data])
            history_df.to_csv(Config.HISTORY_FILE, mode='a', header=False, index=False)
        else:
            history_df = pd.DataFrame([history_data])
            history_df.to_csv(Config.HISTORY_FILE, mode='w', header=True, index=False)
        print(f"Prediksi berhasil disimpan ke history CSV.")

    @staticmethod
    def get_history_from_csv(user_id=None):
        """
        Mengambil history prediksi dari file CSV (opsional, jika masih digunakan paralel dengan DB).
        """
        if os.path.exists(Config.HISTORY_FILE) and os.path.getsize(Config.HISTORY_FILE) > 0:
            history_df = pd.read_csv(Config.HISTORY_FILE)
            if user_id:
                history_df = history_df[history_df['user_id'] == user_id]
            history_df['timestamp_dt'] = pd.to_datetime(history_df['timestamp'])
            history_df = history_df.sort_values(by='timestamp_dt', ascending=False)
            history_df = history_df.drop(columns=['timestamp_dt'])

            history_list = history_df.to_dict(orient='records')
            for item in history_list:
                item['confidence'] = f"{float(item['confidence'])*100:.2f}%"
                item['image_url'] = f"/{item['image_url']}"
            return history_list
        return []

save_prediction_history(...): Metode ini bertanggung jawab untuk menyimpan hasil prediksi ke database.

Ini membuat instansi PredictionHistory baru dengan data yang diberikan.

db.session.add(new_history): Menambahkan objek baru ke sesi database.

db.session.commit(): Menerapkan perubahan ke database. Jika terjadi kesalahan, db.session.rollback() akan membatalkan transaksi untuk menjaga integritas data.

get_user_history(user_id): Mengambil semua riwayat prediksi untuk user_id tertentu.

PredictionHistory.query.filter_by(user_id=user_id).order_by(PredictionHistory.timestamp.desc()).all(): Ini adalah kueri SQLAlchemy yang mengambil semua catatan riwayat yang cocok dengan user_id yang diberikan, diurutkan dari yang terbaru ke yang terlama.

Hasilnya diformat menjadi daftar kamus yang mudah digunakan untuk respons JSON.

get_all_history(): Mengambil semua riwayat prediksi dari database, tidak peduli siapa penggunanya. Ini mirip dengan get_user_history tetapi tanpa filter user_id.

save_prediction_history_to_csv dan get_history_from_csv: Metode ini disediakan jika Anda masih menggunakan penyimpanan riwayat berbasis CSV secara paralel dengan database. Jika Anda sepenuhnya beralih ke database, metode ini bisa dihapus.

Rute API (routes/api_routes.py)
Modul ini mendefinisikan Blueprints dan endpoint API untuk klasifikasi anjing dan riwayat.

# my_dog_breed_classifier/routes/api_routes.py

from flask import Blueprint, request, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
from datetime import datetime
import shutil

# Mengimpor jwt_required dan get_jwt_identity.
# jwt_optional sudah diganti dengan jwt_required(optional=True) di versi terbaru.
from flask_jwt_extended import jwt_required, get_jwt_identity

from config import Config # Mengimpor konfigurasi
from utils.model_loader import get_model, get_idx_to_class_mapping, preprocess_image
from services.classification_service import ClassificationService # Mengimpor layanan klasifikasi

# Membuat instance Blueprint untuk rute API utama
# Semua rute di Blueprint ini akan memiliki prefix '/api'
api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/predict', methods=['POST'])
@jwt_required(optional=True) # PERUBAHAN DI SINI: Prediksi dapat dilakukan oleh guest atau logged-in user
def predict():
    """
    Rute untuk prediksi ras anjing.
    Tidak memerlukan autentikasi JWT secara wajib. Jika token valid disertakan,
    user_id akan disimpan bersama riwayat. Jika tidak ada token atau token tidak valid,
    prediksi akan tetap berjalan, tetapi user_id akan None.
    Menerima file gambar melalui FormData.
    """
    # Mendapatkan ID pengguna dari token JWT jika ada, jika tidak ada, akan mengembalikan None
    current_user_id = get_jwt_identity()

    model = get_model()
    idx_to_class = get_idx_to_class_mapping()

    # Periksa apakah model atau mapping kelas berhasil dimuat
    if model is None or not idx_to_class:
        return jsonify({'error': 'Model AI atau mapping kelas tidak berhasil dimuat di backend.'}), 500

    # Validasi input file gambar
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar yang diunggah.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Nama file kosong, silakan pilih gambar.'}), 400

    temp_filepath = None # Inisialisasi variabel untuk memastikan bisa diakses di blok except
    try:
        # Simpan File yang Diunggah Sementara
        # Mengamankan nama file untuk mencegah Path Traversal
        filename = secure_filename(file.filename)
        # Path lengkap untuk file sementara di folder temp_uploads
        temp_filepath = os.path.join(Config.TEMP_UPLOAD_FOLDER, filename)
        file.save(temp_filepath)

        # Pra-pemrosesan Gambar dan Prediksi
        processed_image = preprocess_image(temp_filepath)
        predictions = model.predict(processed_image)[0]

        predicted_class_index = np.argmax(predictions)
        confidence = float(predictions[predicted_class_index])
        predicted_breed = idx_to_class.get(predicted_class_index, 'Ras Tidak Dikenali')

        # Logika Penyimpanan Gambar History Permanen
        # Buat timestamp dan string acak untuk nama file unik
        timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_filename = f"{timestamp_str}_{os.urandom(4).hex()}_{filename}"
        # Path lengkap untuk menyimpan gambar secara permanen di folder history_images
        history_image_storage_path = os.path.join(Config.HISTORY_IMAGES_FOLDER, unique_filename)

        # Salin file dari folder sementara ke folder history permanen
        shutil.copy(temp_filepath, history_image_storage_path)

        # Hapus file sementara setelah disalin
        os.remove(temp_filepath)

        # Catat Hasil ke History Database Menggunakan Service
        # Path relatif gambar yang akan disimpan di DB dan dikirim ke frontend
        # Ini akan digunakan oleh frontend untuk mengakses gambar via URL '/static/history_images/unique_filename.jpg'
        relative_image_url = f"static/history_images/{unique_filename}"
        ClassificationService.save_prediction_history(
            user_id=current_user_id, # user_id akan berupa None jika guest, atau ID jika login
            image_filename=unique_filename,
            image_url=relative_image_url,
            predicted_breed=predicted_breed,
            confidence=confidence
        )

        # Kirim respons JSON ke frontend
        return jsonify({
            'predicted_breed': predicted_breed,
            'confidence': f"{confidence*100:.2f}%",
            'image_url': f"/{relative_image_url}" # URL lengkap untuk gambar yang disimpan permanen
        }), 200 # OK

    except Exception as e:
        # Penanganan Error
        # Pastikan file sementara dihapus jika terjadi error selama pemprosesan
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        print(f"Error saat memproses permintaan prediksi: {e}")
        return jsonify({'error': f'Terjadi kesalahan saat memproses gambar: {str(e)}'}), 500 # Internal Server Error

# ROUTE UNTUK MENGAMBIL HISTORY KLASIFIKASI (SEKARANG PUBLIK)
@api_bp.route('/history', methods=['GET'])
# @jwt_required() # PERUBAHAN DI SINI: DECORATOR INI DIHAPUS UNTUK AKSES PUBLIK PENUH
def get_history():
    """
    Rute untuk mengambil SEMUA history prediksi ras anjing (publik).
    Tidak memerlukan autentikasi JWT.
    """
    try:
        # Panggil get_all_history untuk mengambil semua data history
        history_list = ClassificationService.get_all_history()
        return jsonify(history_list), 200 # OK
    except Exception as e:
        print(f"Error saat mengambil history: {e}")
        return jsonify({'error': f'Gagal mengambil history: {str(e)}'}), 500 # Internal Server Error

api_bp = Blueprint('api', __name__, url_prefix='/api'): Ini menciptakan Blueprint Flask, sebuah cara untuk mengorganisir rute dan logika terkait ke dalam modul yang lebih kecil. Semua rute yang didefinisikan dalam api_bp akan diawali dengan /api.

@api_bp.route('/predict', methods=['POST']): Ini adalah decorator yang mendefinisikan endpoint API /api/predict yang hanya menerima permintaan POST.

@jwt_required(optional=True): Decorator ini dari Flask-JWT-Extended menandakan bahwa endpoint ini dapat diakses oleh pengguna yang diautentikasi (jika mereka menyertakan token JWT yang valid) atau oleh pengguna tamu (jika tidak ada token atau token tidak valid). get_jwt_identity() akan mengembalikan ID pengguna jika token valid, atau None jika tidak.

Penanganan Unggahan File: Kode ini memeriksa apakah ada file yang diunggah ('file' in request.files) dan apakah nama file tidak kosong. secure_filename() digunakan untuk membersihkan nama file untuk mencegah masalah keamanan. File sementara disimpan di Config.TEMP_UPLOAD_FOLDER.

Prediksi Model: Gambar yang diunggah diproses menggunakan preprocess_image() dari model_loader.py. Kemudian, model.predict() digunakan untuk mendapatkan probabilitas kelas. np.argmax() menemukan kelas dengan probabilitas tertinggi, dan tingkat kepercayaan diekstrak.

Penyimpanan Gambar Riwayat: Gambar yang diunggah disalin ke Config.HISTORY_IMAGES_FOLDER dengan nama file unik (menggunakan timestamp dan string acak) untuk penyimpanan permanen. File sementara kemudian dihapus.

ClassificationService.save_prediction_history(...): Memanggil metode dari layanan klasifikasi untuk menyimpan detail prediksi (termasuk user_id, nama file gambar, URL, ras yang diprediksi, dan kepercayaan) ke database.

Respons JSON: Mengembalikan hasil prediksi (ras, kepercayaan, URL gambar) sebagai respons JSON.

Penanganan Error: Blok try...except menangkap pengecualian selama proses prediksi, menghapus file sementara jika terjadi kesalahan, dan mengembalikan respons error JSON yang sesuai.

@api_bp.route('/history', methods=['GET']): Mendefinisikan endpoint API /api/history yang menerima permintaan GET.

ClassificationService.get_all_history(): Memanggil metode dari layanan klasifikasi untuk mengambil semua riwayat prediksi dari database.

Respons JSON Riwayat: Mengembalikan daftar riwayat sebagai respons JSON.

Rute Utama (routes/main_routes.py)
Modul ini mendefinisikan Blueprint dan rute utama untuk menyajikan halaman HTML utama.

from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__, template_folder='../templates')

@main_bp.route('/')
def index():
    return render_template('index.html')

main_bp = Blueprint('main', __name__, template_folder='../templates'): Ini menciptakan Blueprint Flask bernama main. template_folder='../templates' memberi tahu Flask di mana menemukan file templat HTML untuk Blueprint ini.

@main_bp.route('/'): Ini adalah decorator yang mengaitkan fungsi index() dengan URL root (/) aplikasi.

return render_template('index.html'): Fungsi ini akan mencari file index.html di dalam folder templates (yang ditentukan oleh template_folder) dan mengirimkannya sebagai respons ke klien web. Ini adalah cara Flask menyajikan halaman frontend statis.

Titik Masuk Flask CLI (run.py)
File run.py berfungsi sebagai titik masuk untuk perintah flask CLI (Command Line Interface), seperti flask run atau flask db migrate.

# my_dog_breed_classifier/run.py

# File ini adalah titik masuk untuk perintah 'flask' CLI.
# FLASK_APP yang diatur ke 'run.py' akan menemukan fungsi 'create_app()'.

from app import create_app # Import fungsi create_app dari app.py
from extensions import db # Import instance 'db' dari extensions.py
from models_db.user import User # Import model-model Anda untuk shell context
from models_db.prediction_history import PredictionHistory # Import model-model Anda untuk shell context

app = create_app()

# Menambahkan User dan PredictionHistory ke shell context
# Ini berguna jika Anda ingin berinteraksi dengan database dari Flask shell (flask shell)
@app.shell_context_processor
def make_shell_context():
    return dict(app=app, db=db, User=User, PredictionHistory=PredictionHistory)

# Anda tidak perlu baris 'if __name__ == '__main__': app.run(...)' di sini
# karena Anda akan menggunakan perintah 'flask run' dari terminal.

from app import create_app: Mengimpor fungsi create_app dari app.py. Ini adalah bagaimana Flask CLI menemukan aplikasi Anda.

app = create_app(): Membuat instansi aplikasi Flask.

@app.shell_context_processor: Ini adalah decorator yang mendaftarkan fungsi make_shell_context(). Fungsi ini mengembalikan kamus objek yang akan tersedia secara otomatis saat Anda menjalankan flask shell. Ini sangat berguna untuk debugging dan berinteraksi dengan database dari baris perintah, karena Anda dapat langsung mengakses db, User, dan PredictionHistory tanpa perlu mengimpornya secara manual di shell.

Kontribusi
Kontribusi dipersilakan! Silakan buka issue atau kirim pull request.

Lisensi
Proyek ini dilisensikan di bawah Lisensi MIT.

