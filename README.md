# 🐶 Dog Breed Classifier Backend

Aplikasi backend berbasis **Flask** untuk mengklasifikasikan ras anjing dari gambar menggunakan model **Convolutional Neural Network (CNN)**. Sistem ini terintegrasi dengan **MySQL** untuk menyimpan riwayat prediksi, mendukung otentikasi JWT, dan menyediakan RESTful API.

---

## 📋 Daftar Isi

- [✨ Fitur](#-fitur)
- [📁 Struktur Proyek](#-struktur-proyek)
- [🛠 Teknologi yang Digunakan](#-teknologi-yang-digunakan)
- [⚙️ Pengaturan dan Instalasi](#-pengaturan-dan-instalasi)
  - [📌 Prasyarat](#-prasyarat)
  - [📂 Pengaturan Database](#-pengaturan-database)
  - [🔐 Variabel Lingkungan](#-variabel-lingkungan)
  - [📦 Instalasi Dependensi](#-instalasi-dependensi)
  - [📤 Migrasi Database](#-migrasi-database)
  - [📚 Menyiapkan Model AI](#-menyiapkan-model-ai)
- [🚀 Menjalankan Aplikasi](#-menjalankan-aplikasi)
- [📡 API Endpoint](#-api-endpoint)
- [🧠 Penjelasan Program](#-penjelasan-program)
- [🤝 Kontribusi](#-kontribusi)
- [📄 Lisensi](#-lisensi)

---

## ✨ Fitur

- ✅ Klasifikasi ras anjing dari gambar menggunakan **ResNet50V2**
- 📊 Penyimpanan riwayat prediksi lengkap ke database
- 🔐 Otentikasi berbasis **JWT (opsional)**
- 🔄 **CORS** support untuk integrasi frontend
- 📸 Penyimpanan gambar sementara dan permanen
- 🧱 Struktur modular menggunakan Flask Blueprints

---

## 📁 Struktur Proyek

```bash
my_dog_breed_classifier/
├── app.py
├── config.py
├── extensions.py
├── run.py
├── dog-breed-classifier.ipynb
├── models/
│   ├── dog_breed_classifier_final_model.keras
│   └── class_indices.json
├── models_db/
│   ├── prediction_history.py
│   └── user.py
├── routes/
│   ├── api_routes.py
│   └── main_routes.py
├── services/
│   └── classification_service.py
├── static/
│   ├── history_images/
│   └── temp_uploads/
├── templates/
│   └── index.html
├── data/
└── migrations/
