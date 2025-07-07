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


