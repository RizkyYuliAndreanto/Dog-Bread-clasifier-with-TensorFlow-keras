# ✨ Dog Bread Clasifier With TensorFlow Keras

[![Language](https://img.shields.io/github/languages/top/RizkyYuliAndreanto/Dog-Bread-clasifier-with-TensorFlow-keras?style=flat-square)](https://github.com/RizkyYuliAndreanto/Dog-Bread-clasifier-with-TensorFlow-keras)
[![Python](https://img.shields.io/badge/python-3.8+-blue?style=flat-square)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-%23000.svg?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/en/2.3.x/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-%23FF6F00.svg?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)


> Aplikasi backend berbasis Flask untuk mengklasifikasikan ras anjing dari gambar menggunakan model Convolutional Neural Network (CNN) yang telah dilatih.  Aplikasi ini terintegrasi dengan database MySQL untuk menyimpan riwayat prediksi dan menyediakan API endpoint untuk klasifikasi gambar dan pengambilan data historis.

## ✨ Fitur Utama

* **Klasifikasi Ras Anjing:** Memprediksi ras anjing dari gambar yang diunggah menggunakan model CNN pra-terlatih (ResNet50V2).
* **Riwayat Prediksi:** Menyimpan hasil klasifikasi (gambar, ras yang diprediksi, tingkat kepercayaan, dan ID pengguna) dalam database MySQL.
* **Otentikasi Pengguna (Opsional):**  Menggunakan Flask-JWT-Extended untuk otentikasi pengguna opsional pada endpoint prediksi. Pengguna tamu juga dapat melakukan prediksi.
* **CORS Enabled:** Dikonfigurasi untuk menangani Cross-Origin Resource Sharing untuk integrasi frontend yang mulus.
* **Migrasi Database:** Menggunakan Flask-Migrate untuk pengelolaan skema database yang mudah.
* **Struktur Modular:**  Diatur ke dalam Blueprints, layanan, dan modul utilitas untuk pemeliharaan yang lebih baik.
* **Penyimpanan Gambar:** Menyimpan gambar yang diunggah untuk diproses dan menyimpannya secara permanen untuk melihat riwayat.


## 🛠️ Tumpukan Teknologi

| Kategori             | Teknologi        | Catatan                                          |
|----------------------|--------------------|------------------------------------------------------|
| Bahasa Pemrograman   | Python            | Versi 3.8+ direkomendasikan                         |
| Framework Web        | Flask             | Untuk membangun backend API                         |
| Database             | MySQL             | Menyimpan riwayat prediksi                           |
| ORM                  | SQLAlchemy        | Object-Relational Mapper untuk interaksi database     |
| Migrasi Database     | Flask-Migrate      | Untuk pengelolaan skema database                       |
| Library Deep Learning | TensorFlow/Keras | Untuk membangun dan menjalankan model CNN            |
| Otentikasi           | Flask-JWT-Extended | Untuk otentikasi pengguna (opsional)                 |
| CORS                 | Flask-CORS        | Untuk menangani permintaan cross-origin                |
| Lain-lain            | NumPy, Pillow, Pandas, scikit-learn, Matplotlib, python-dotenv, Werkzeug | Untuk manipulasi data, pemrosesan gambar, dan visualisasi |


## 🏛️ Tinjauan Arsitektur

Aplikasi ini mengikuti arsitektur tiga lapis (tiga tier) yang umum:

1. **Presentasi (Frontend):**  (Tidak termasuk dalam repositori ini,  frontend akan mengunggah gambar dan menampilkan hasil.)
2. **Aplikasi (Backend):**  Dibangun menggunakan Flask, menangani rute, logika bisnis, dan interaksi dengan database.
3. **Data:**  Data disimpan dalam database MySQL menggunakan SQLAlchemy ORM.


## 🚀 Memulai

1. **Kloning Repositori:**
   ```bash
   git clone https://github.com/RizkyYuliAndreanto/Dog-Bread-clasifier-with-TensorFlow-keras.git
   cd Dog-Bread-clasifier-with-TensorFlow-keras
   ```

2. **Buat dan konfigurasi database MySQL:**  Ikuti instruksi di `README.md` untuk membuat database `dog_classifier_db`.  Pastikan Anda menyesuaikan `DATABASE_URL` di file `.env` (atau `config.py` jika tidak menggunakan `.env`).

3. **Instal Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan Migrasi:**
   ```bash
   flask db init
   flask db migrate -m "Initial migration"
   flask db upgrade
   ```

5. **Latih Model (Jika perlu):** Jalankan `dog-breed-classifier.ipynb` di lingkungan Jupyter Notebook untuk melatih model.  Unduh `dog_breed_classifier_final_model.keras` dan `class_indices.json` dan letakkan di direktori `models/`.

6. **Jalankan Aplikasi:**
   ```bash
   export FLASK_APP=run.py  # Untuk Linux/macOS
   flask run
   ```
   (Atau `set FLASK_APP=run.py` dan `flask run` di Windows.)


## 📂 Struktur File

```
/
├── .flaskenv
├── .gitattributes
├── .gitignore
├── README.md
├── app.py             # Application factory untuk Flask
├── config.py          # Konfigurasi aplikasi (database, path, kunci rahasia)
├── dog-breed-classifier.ipynb # Jupyter Notebook untuk pelatihan model
├── extensions.py       # Inisialisasi ekstensi Flask (SQLAlchemy, JWT, CORS, Migrate)
├── migrations         # Direktori untuk skrip migrasi database
├── models_db          # Model database SQLAlchemy
├── requirements.txt   # Daftar dependensi Python
├── routes             # Blueprints Flask untuk rute API dan utama
├── run.py             # Titik masuk untuk perintah Flask CLI
├── services           # Layanan logika bisnis (klasifikasi, otentikasi)
├── static             # File statis (gambar, CSS, JS)
│   └── history_images
├── templates          # Templat HTML
└── utils              # Utilitas (pemuatan model, pra-pemrosesan gambar)
```

* **`app.py`:**  Inti aplikasi Flask.  Menggunakan pattern application factory.
* **`routes/`:** Mengandung Blueprints Flask untuk mengorganisir rute API dan rute utama.
* **`models_db/`:**  Mendefinisikan model database SQLAlchemy.
* **`services/`:**  Berisi logika bisnis yang terkait dengan klasifikasi dan pengelolaan riwayat prediksi.
* **`utils/`:**  Berisi fungsi utilitas seperti memuat model dan pra-pemrosesan gambar.
* **`static/`:**  Berisi aset statis seperti gambar.
* **`templates/`:**  Berisi templat HTML (hanya `index.html` dalam contoh ini).


# Dog Breed Classifier Backend

Proyek ini adalah bagian *backend* dari aplikasi klasifikasi ras anjing yang dibangun menggunakan Flask. Backend ini bertanggung jawab untuk mengatur konfigurasi aplikasi, manajemen database, otentikasi pengguna menggunakan JWT, dan penyiapan *endpoints* API untuk interaksi dengan model klasifikasi.

## Penjelasan Kode dengan Komentar Inline

Berikut adalah penjelasan mendalam mengenai fungsi dan isi dari setiap file kode yang Anda berikan, dengan komentar langsung di dalam kode:

---

### 1. `config.py`

File ini mendefinisikan kelas `Config` yang berisi semua pengaturan global dan *path* penting untuk aplikasi Flask. Ini adalah pusat konfigurasi aplikasi.

```python
import os # Mengimpor modul os untuk berinteraksi dengan sistem operasi, terutama untuk manipulasi path file.
from datetime import timedelta # Mengimpor timedelta dari modul datetime untuk mengatur durasi waktu, digunakan untuk masa berlaku token JWT.

class Config:
    """
    Kelas konfigurasi untuk aplikasi Flask.
    Menyimpan semua path penting dan pengaturan global.
    """
    # Mendapatkan path absolut ke direktori dasar proyek (tempat file config.py berada)
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    # BASE_DIR akan menyimpan path absolut ke direktori tempat 'config.py' berada.

    # Kunci rahasia untuk keamanan aplikasi Flask (misalnya untuk sesi).
    # Diambil dari variabel lingkungan 'SECRET_KEY' jika ada, jika tidak menggunakan nilai default.
    SECRET_KEY = os.environ.get('SECRET_KEY', '1a2b3c4d5e6f7g8h9i0j11213141516171819202122232425')
    DEBUG = True # Mengaktifkan mode debug, berguna untuk pengembangan karena memberikan informasi error yang lebih detail dan reload otomatis.
    HOST = '0.0.0.0' # Mengatur host aplikasi agar dapat diakses dari IP manapun (bukan hanya localhost).
    PORT = 5000 # Mengatur port di mana aplikasi akan berjalan.

    # --- Konfigurasi Database (MySQL via PyMySQL) ---
    # String koneksi ke database MySQL Anda.
    # Diambil dari variabel lingkungan (DATABASE_URL) atau nilai default untuk pengembangan.
    # Ganti 'dog_classifier_db' dengan nama database yang Anda buat di Laragon.
    # Ganti 'root' dan '' jika Anda mengubah user/password default Laragon.
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'mysql+pymysql://root:@localhost:3306/dog_classifier_db'
    # SQLALCHEMY_DATABASE_URI adalah string koneksi untuk Flask-SQLAlchemy ke database MySQL.
    # Jika variabel lingkungan DATABASE_URL tidak diset, akan menggunakan default untuk Laragon.
    SQLALCHEMY_TRACK_MODIFICATIONS = False # Menonaktifkan pelacakan modifikasi objek SQLAlchemy, direkomendasikan untuk performa.

    # Kunci rahasia untuk JWT (JSON Web Tokens) yang digunakan untuk otentikasi pengguna.
    # Diambil dari variabel lingkungan 'JWT_SECRET_KEY' jika ada, jika tidak menggunakan nilai default.
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'a9b8c7d6e5f4g3h2i1j0192837465abcd1234567890abcdef')
    # Mengatur masa berlaku (kadaluarsa) untuk token akses JWT menjadi 3 hari.
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=3) # <-- DISINI KADALUARSA TOKEN DIATUR KE 1 JAM (Komentar asli menyebut 1 jam, tapi kodenya diatur ke 3 hari. Ini perlu disesuaikan jika ada inkonsistensi)
    # JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30) # Opsional: untuk refresh token yang lebih lama (dikomentari, tidak aktif)

    # --- Path untuk Model AI ---
    # Mendefinisikan path lengkap ke file model klasifikasi ras anjing yang telah dilatih.
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'dog_breed_classifier_final_model.keras')
    # Mendefinisikan path lengkap ke file JSON yang berisi pemetaan indeks kelas ke nama ras anjing.
    CLASS_INDICES_PATH = os.path.join(BASE_DIR, 'models', 'class_indices.json')

    # --- UKURAN GAMBAR ---
    # Ukuran tinggi gambar input yang diharapkan oleh model CNN (Convolutional Neural Network).
    IMG_HEIGHT = 224
    # Ukuran lebar gambar input yang diharapkan oleh model CNN.
    IMG_WIDTH = 224

    # --- Path untuk Folder Unggahan Gambar dan History Gambar ---
    # Path ke folder sementara tempat gambar yang diunggah oleh pengguna akan disimpan sebelum diproses.
    TEMP_UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'temp_uploads')
    # Path ke folder permanen tempat gambar yang telah diklasifikasikan akan disimpan sebagai riwayat.
    HISTORY_IMAGES_FOLDER = os.path.join(BASE_DIR, 'static', 'history_images')

    # --- Path untuk History Klasifikasi (jika masih menggunakan CSV paralel) ---
    # Jika Anda sepenuhnya beralih ke database, bagian ini bisa diabaikan atau dihapus
    # Path ke folder untuk menyimpan file riwayat klasifikasi (misalnya, file CSV).
    HISTORY_FOLDER = os.path.join(BASE_DIR, 'data')
    # Path lengkap ke file CSV yang menyimpan riwayat klasifikasi.
    HISTORY_FILE = os.path.join(HISTORY_FOLDER, 'classification_history.csv')



## 2.config.py

```python
import os # Mengimpor modul 'os' yang menyediakan cara untuk berinteraksi dengan sistem operasi, seperti manipulasi jalur file dan direktori.
from datetime import timedelta # Mengimpor 'timedelta' dari modul 'datetime' untuk merepresentasikan durasi waktu, yang akan digunakan untuk mengatur masa berlaku token.

class Config:
    """
    Kelas konfigurasi untuk aplikasi Flask.
    Kelas ini berfungsi sebagai wadah terpusat untuk semua pengaturan penting dan jalur file global yang dibutuhkan oleh aplikasi Flask.
    """
    # Mendapatkan path absolut ke direktori dasar proyek (tempat file config.py berada)
    # os.path.dirname(__file__) mendapatkan direktori dari file saat ini (config.py).
    # os.path.abspath() mengubah jalur relatif menjadi jalur absolut.
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    # Kunci rahasia (secret key) untuk keamanan aplikasi Flask.
    # Digunakan untuk menandatangani cookie sesi dan untuk tujuan keamanan lainnya.
    # Nilainya diambil dari variabel lingkungan 'SECRET_KEY' jika ada, jika tidak, menggunakan nilai default yang panjang dan acak.
    SECRET_KEY = os.environ.get('SECRET_KEY', '1a2b3c4d5e6f7g8h9i0j11213141516171819202122232425')
    
    DEBUG = True # Mengatur aplikasi ke mode debug. Dalam mode debug, aplikasi akan memberikan informasi error yang lebih rinci dan akan otomatis me-reload saat ada perubahan kode. Ini berguna untuk pengembangan.
    HOST = '0.0.0.0' # Mengatur host di mana aplikasi akan berjalan. '0.0.0.0' berarti aplikasi akan dapat diakses dari semua alamat IP yang tersedia di mesin, bukan hanya localhost.
    PORT = 5000 # Mengatur nomor port di mana aplikasi akan mendengarkan permintaan masuk.

    # --- Konfigurasi Database (MySQL via PyMySQL) ---
    # String koneksi (URI) untuk database SQLAlchemy.
    # Ini menentukan jenis database (mysql), driver (pymysql), kredensial (root:@), host (localhost:3306), dan nama database (dog_classifier_db).
    # Nilainya diambil dari variabel lingkungan 'DATABASE_URL' (umumnya untuk deployment), atau menggunakan nilai default untuk pengembangan lokal.
    # Anda mungkin perlu mengganti 'dog_classifier_db', 'root', dan '' sesuai dengan konfigurasi database MySQL Anda (misalnya di Laragon).
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'mysql+pymysql://root:@localhost:3306/dog_classifier_db'
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False # Menonaktifkan fitur pelacakan modifikasi objek SQLAlchemy. Ini direkomendasikan karena dapat menghemat memori dan tidak diperlukan untuk sebagian besar aplikasi.

    # Kunci rahasia untuk Json Web Tokens (JWT).
    # Digunakan untuk menandatangani dan memverifikasi JWT yang digunakan untuk otentikasi pengguna.
    # Diambil dari variabel lingkungan 'JWT_SECRET_KEY' atau menggunakan nilai default.
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'a9b8c7d6e5f4g3h2i1j0192837465abcd1234567890abcdef')
    
    # Mengatur masa berlaku (expiration time) untuk token akses JWT.
    # timedelta(days=3) berarti token akan kadaluarsa setelah 3 hari.
    # Komentar asli menyebut "1 JAM", namun kode mengaturnya ke 3 hari. Pastikan ini sesuai dengan kebutuhan Anda.
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=3) 
    
    # JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30) # Opsional: Mengatur masa berlaku untuk refresh token. Baris ini dikomentari, jadi tidak aktif.

    # --- Path untuk Model AI ---
    # Menggabungkan BASE_DIR dengan 'models' dan nama file model untuk mendapatkan jalur absolut ke model AI.
    # Model ini kemungkinan adalah model Keras yang telah dilatih untuk klasifikasi ras anjing.
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'dog_breed_classifier_final_model.keras')
    
    # Menggabungkan BASE_DIR dengan 'models' dan nama file JSON untuk mendapatkan jalur absolut ke file indeks kelas.
    # File ini kemungkinan berisi pemetaan antara indeks numerik yang dihasilkan oleh model dengan nama ras anjing yang sebenarnya.
    CLASS_INDICES_PATH = os.path.join(BASE_DIR, 'models', 'class_indices.json')

    # --- UKURAN GAMBAR ---
    # Menentukan tinggi gambar input yang diharapkan oleh model CNN (Convolutional Neural Network).
    # Gambar yang akan diproses oleh model harus diubah ukurannya menjadi dimensi ini.
    IMG_HEIGHT = 224
    # Menentukan lebar gambar input yang diharapkan oleh model CNN.
    IMG_WIDTH = 224

    # --- Path untuk Folder Unggahan Gambar dan History Gambar ---
    # Path ke folder sementara di mana gambar yang baru diunggah oleh pengguna akan disimpan.
    # Gambar-gambar ini akan diproses untuk klasifikasi sebelum mungkin dipindahkan atau dihapus.
    TEMP_UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'temp_uploads')
    
    # Path ke folder permanen di mana gambar yang telah diklasifikasikan (sebagai bagian dari riwayat prediksi) akan disimpan.
    HISTORY_IMAGES_FOLDER = os.path.join(BASE_DIR, 'static', 'history_images')

    # --- Path untuk History Klasifikasi (jika masih menggunakan CSV paralel) ---
    # Bagian ini mungkin relevan jika ada sistem riwayat klasifikasi berbasis CSV yang berjalan paralel dengan atau sebelum migrasi penuh ke database.
    # Jika Anda sepenuhnya beralih ke database, bagian ini bisa diabaikan atau dihapus.
    
    # Path ke folder tempat file riwayat klasifikasi (misalnya, file CSV) akan disimpan.
    HISTORY_FOLDER = os.path.join(BASE_DIR, 'data')
    
    # Path lengkap ke file CSV spesifik yang menyimpan riwayat klasifikasi.
    HISTORY_FILE = os.path.join(HISTORY_FOLDER, 'classification_history.csv')



## 3. run.py

```python
# my_dog_breed_classifier/run.py

# ==============================================================================
# PENJELASAN UMUM:
# File ini berfungsi sebagai titik masuk (entry point) utama untuk aplikasi Flask
# ketika dijalankan menggunakan perintah 'flask' CLI (Command Line Interface).
# Ini adalah praktik standar dalam pengembangan Flask yang modular.
#
# CARA KERJA:
# Ketika Anda menjalankan 'flask run' di terminal, Flask secara otomatis
# mencari file yang diatur sebagai FLASK_APP (dalam kasus ini, 'run.py').
# Kemudian, ia akan mencari dan menjalankan fungsi 'create_app()' di dalam
# file tersebut untuk mendapatkan instance aplikasi Flask yang sudah dikonfigurasi.
#
# TUJUAN UTAMA:
# 1. Mengimpor fungsi 'create_app' dari 'app.py' untuk membuat aplikasi.
# 2. Menyiapkan 'shell context' agar model database dan objek penting lainnya
#    dapat diakses dengan mudah saat menggunakan 'flask shell'.
# ==============================================================================

# ==============================================================================
# IMPOR MODUL:
# Bagian ini mengimpor semua dependensi yang diperlukan dari modul lain
# dalam struktur proyek.
# ==============================================================================
from app import create_app # Mengimpor fungsi 'create_app' dari modul 'app.py'. Fungsi ini adalah 'factory' yang menghasilkan instance aplikasi Flask yang sudah dikonfigurasi.
from extensions import db # Mengimpor instance 'db' dari modul 'extensions.py'. 'db' biasanya adalah objek Flask-SQLAlchemy yang merepresentasikan koneksi dan fungsionalitas database.
from models_db.user import User # Mengimpor model 'User' dari 'models_db/user.py'. Model ini merepresentasikan tabel pengguna di database dan akan ditambahkan ke shell context.
from models_db.prediction_history import PredictionHistory # Mengimpor model 'PredictionHistory' dari 'models_db/prediction_history.py'. Model ini merepresentasikan tabel riwayat prediksi di database dan akan ditambahkan ke shell context.

# ==============================================================================
# INISIALISASI APLIKASI:
# Memanggil fungsi 'create_app' untuk membuat instance aplikasi Flask.
# ==============================================================================
app = create_app() # Instance aplikasi Flask yang sudah dikonfigurasi sepenuhnya kini disimpan dalam variabel 'app'.

# ==============================================================================
# KONFIGURASI SHELL CONTEXT:
# Bagian ini mendaftarkan objek-objek penting (aplikasi, database, model)
# ke dalam Flask shell context. Ini memungkinkan pengembang untuk berinteraksi
# langsung dengan aplikasi dan database dari command line.
# ==============================================================================
@app.shell_context_processor # Dekorator ini mendaftarkan fungsi 'make_shell_context' sebagai pembuat konteks shell. Flask akan memanggil fungsi ini saat 'flask shell' dimulai.
def make_shell_context():
    """
    Fungsi ini mengembalikan dictionary objek-objek yang akan tersedia secara otomatis
    di Flask shell. Dengan ini, Anda bisa langsung mengakses objek-objek seperti
    'app', 'db', 'User', atau 'PredictionHistory' tanpa perlu mengimpornya lagi
    setiap kali Anda masuk ke shell. Ini sangat membantu untuk debugging dan
    interaksi database.
    """
    return dict(app=app, db=db, User=User, PredictionHistory=PredictionHistory) # Mengembalikan dictionary objek yang akan tersedia di shell.

# ==============================================================================
# CATATAN PENTING:
# Tidak diperlukan blok 'if __name__ == '__main__': app.run(...)' di sini.
# ==============================================================================
# Karena file ini berfungsi sebagai entry point untuk 'flask' CLI, Flask secara otomatis
# akan menemukan fungsi 'create_app()' dan menjalankan aplikasi. Baris 'app.run()'
# hanya diperlukan jika Anda ingin menjalankan aplikasi sebagai skrip Python biasa
# (misalnya 'python run.py'), yang bukan merupakan praktik terbaik untuk aplikasi Flask
# yang lebih besar. Penggunaan 'flask run' lebih disukai.


## 4. model_loader.py
```python
import tensorflow as tf # Mengimpor library TensorFlow, kerangka kerja open-source untuk pembelajaran mesin.
from tensorflow.keras.preprocessing import image # Mengimpor modul 'image' dari Keras preprocessing untuk fungsi pengolahan gambar seperti konversi gambar ke array NumPy.
import numpy as np # Mengimpor library NumPy, fundamental untuk komputasi numerik di Python, terutama untuk array.
import json # Mengimpor modul 'json' untuk bekerja dengan data JSON, khususnya untuk memuat mapping kelas.
import os # Mengimpor modul 'os' untuk berinteraksi dengan sistem operasi, seperti manajemen path dan pembuatan direktori.
from PIL import Image # Mengimpor kelas 'Image' dari Pillow (PIL Fork), library pengolahan gambar yang powerful.

# Import kelas Config dari file config.py (Import absolut yang benar)
# Ini memungkinkan akses ke pengaturan global seperti path model, path indeks kelas, dan ukuran gambar.
from config import Config

# ==============================================================================
# VARIABEL GLOBAL UNTUK MODEL DAN MAPPING KELAS
# ==============================================================================
# Variabel global ini akan menyimpan instance model TensorFlow/Keras yang sudah dimuat
# dan mapping indeks kelas ke nama ras anjing.
# Tujuannya adalah untuk memastikan model hanya dimuat sekali saat aplikasi dimulai
# (singleton pattern), menghindari pemuatan ulang yang memakan waktu pada setiap permintaan prediksi.
_model = None # Variabel untuk menyimpan objek model Keras yang akan dimuat. Diinisialisasi sebagai None.
_idx_to_class = None # Variabel untuk menyimpan dictionary mapping dari indeks numerik ke nama kelas ras anjing. Diinisialisasi sebagai None.

# ==============================================================================
# FUNGSI UNTUK MEMUAT MODEL DAN MAPPING KELAS
# ==============================================================================
def load_model_and_class_indices():
    """
    Memuat model Keras dan mapping kelas ke memori.
    Fungsi ini dirancang untuk dipanggil saat modul ini pertama kali diimpor,
    memastikan bahwa model dan data mapping kelas siap digunakan sejak aplikasi dimulai.
    Ini adalah langkah inisialisasi kritis.
    """
    global _model, _idx_to_class # Mendeklarasikan bahwa kita akan memodifikasi variabel global _model dan _idx_to_class.

    # Muat model hanya jika belum dimuat sebelumnya (_model masih None)
    if _model is None:
        try:
            _model = tf.keras.models.load_model(Config.MODEL_PATH) # Memuat model Keras dari path yang ditentukan di Config.
            print(f"Model AI berhasil dimuat dari: {Config.MODEL_PATH}") # Pesan konfirmasi jika model berhasil dimuat.
            # Opsional: pesan bahwa model dimuat ke CPU/GPU
            if tf.config.list_physical_devices('GPU'): # Memeriksa apakah ada GPU yang tersedia dan digunakan oleh TensorFlow.
                print("Model dimuat untuk inferensi GPU.") # Pesan jika model akan menggunakan GPU.
            else:
                print("Model dimuat untuk inferensi CPU.") # Pesan jika model akan menggunakan CPU.
        except Exception as e: # Menangani error jika proses pemuatan model gagal.
            print(f"ERROR: Gagal memuat model dari {Config.MODEL_PATH}. Pastikan file ada dan benar. Error: {e}") # Pesan error yang informatif.
            _model = None # Mengatur _model kembali ke None jika gagal, menandakan model tidak tersedia.

    # Muat mapping kelas hanya jika belum dimuat sebelumnya (_idx_to_class masih None)
    if _idx_to_class is None:
        try:
            with open(Config.CLASS_INDICES_PATH, 'r') as f: # Membuka file JSON yang berisi indeks kelas dalam mode baca.
                class_indices_raw = json.load(f) # Memuat konten JSON ke dalam dictionary.
            # Kunci dari JSON biasanya string (misal "0", "1"), konversi ke integer karena indeks kelas adalah integer.
            _idx_to_class = {int(k): v for k, v in class_indices_raw.items()} # Membuat dictionary baru dengan kunci integer.
            print(f"Mapping kelas berhasil dimuat dari: {Config.CLASS_INDICES_PATH}") # Pesan konfirmasi jika mapping berhasil dimuat.
        except Exception as e: # Menangani error jika proses pemuatan mapping kelas gagal.
            print(f"ERROR: Gagal memuat mapping kelas dari {Config.CLASS_INDICES_PATH}. Error: {e}") # Pesan error yang informatif.
            _idx_to_class = {} # Mengatur _idx_to_class ke dictionary kosong jika gagal, menandakan mapping tidak tersedia.

# ==============================================================================
# FUNGSI GETTER UNTUK MODEL DAN MAPPING
# ==============================================================================
# Fungsi-fungsi ini memungkinkan modul lain untuk mengakses instance model
# dan mapping kelas yang sudah dimuat secara aman.
def get_model():
    """Mengembalikan instance model TensorFlow/Keras yang sudah dimuat."""
    # Jika model belum dimuat (misal karena error awal atau belum diinisialisasi), coba muat lagi.
    if _model is None:
        load_model_and_class_indices() # Memanggil fungsi pemuatan jika model belum tersedia.
    return _model # Mengembalikan objek model.

def get_idx_to_class_mapping():
    """Mengembalikan mapping kelas (indeks ke nama ras) yang sudah dimuat."""
    # Jika mapping belum dimuat, coba muat lagi.
    if _idx_to_class is None:
        load_model_and_class_indices() # Memanggil fungsi pemuatan jika mapping belum tersedia.
    return _idx_to_class # Mengembalikan dictionary mapping kelas.

# ==============================================================================
# FUNGSI PRA-PEMROSESAN GAMBAR
# ==============================================================================
def preprocess_image(image_path):
    """
    Pra-pemrosesan gambar agar sesuai dengan format input yang diharapkan oleh model CNN.
    Langkah-langkah pra-pemrosesan ini harus identik dengan pra-pemrosesan yang dilakukan
    saat model dilatih untuk memastikan konsistensi dan akurasi prediksi.
    """
    # Membuka gambar dari path menggunakan Pillow dan memastikan dikonversi ke format RGB (3 channel warna).
    img = Image.open(image_path).convert('RGB')
    # Mengubah ukuran gambar sesuai dengan dimensi input yang diharapkan oleh model (misal 224x224 piksel).
    img = img.resize((Config.IMG_WIDTH, Config.IMG_HEIGHT))
    # Mengonversi objek gambar Pillow menjadi array NumPy.
    img_array = image.img_to_array(img)
    # Menambahkan dimensi batch di awal array.
    # Model Keras/TensorFlow mengharapkan input dalam bentuk batch (misal (batch_size, height, width, channels)),
    # meskipun hanya ada satu gambar, dimensi batch_size (1) harus ditambahkan.
    img_array = np.expand_dims(img_array, axis=0)
    # Normalisasi nilai piksel dari rentang [0, 255] ke [0, 1].
    # Ini dilakukan dengan membagi setiap nilai piksel dengan 255.0.
    # Langkah ini SANGAT PENTING dan HARUS SAMA PERSIS dengan normalisasi yang diterapkan saat pelatihan model!
    img_array = img_array / 255.0
    return img_array # Mengembalikan array NumPy gambar yang sudah diproses dan siap untuk diprediksi oleh model.

# ==============================================================================
# INISIALISASI OTOMATIS SAAT MODUL DIIMPOR
# ==============================================================================
# Bagian kode ini akan dieksekusi secara otomatis satu kali saat modul ini
# (misalnya, `model_loader.py`) pertama kali diimpor ke dalam aplikasi Flask.
# Ini memastikan bahwa folder yang dibutuhkan sudah ada dan model AI dimuat
# ke memori sebelum ada permintaan prediksi.

# Membuat folder sementara untuk unggahan gambar jika belum ada.
if not os.path.exists(Config.TEMP_UPLOAD_FOLDER):
    os.makedirs(Config.TEMP_UPLOAD_FOLDER)

# Membuat folder permanen untuk gambar riwayat klasifikasi jika belum ada.
if not os.path.exists(Config.HISTORY_IMAGES_FOLDER):
    os.makedirs(Config.HISTORY_IMAGES_FOLDER)

# Membuat folder untuk file CSV riwayat (walaupun riwayat utama disimpan di DB,
# folder ini mungkin masih digunakan untuk backup, debugging, atau tujuan lain).
if not os.path.exists(Config.HISTORY_FOLDER):
    os.makedirs(Config.HISTORY_FOLDER)

# Memuat model AI dan mapping kelas saat modul ini pertama kali diimpor.
# Ini memastikan model siap segera setelah aplikasi Flask dimulai,
# mengurangi latensi pada permintaan prediksi pertama.
load_model_and_class_indices()
