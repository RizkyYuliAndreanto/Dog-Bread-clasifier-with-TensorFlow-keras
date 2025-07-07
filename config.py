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

    # --- Konfigurasi Database (MySQL via PyMySQL) ---
    # String koneksi ke database MySQL Anda.
    # Diambil dari variabel lingkungan (DATABASE_URL) atau nilai default untuk pengembangan.
    # Ganti 'dog_classifier_db' dengan nama database yang Anda buat di Laragon.
    # Ganti 'root' dan '' jika Anda mengubah user/password default Laragon.
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'mysql+pymysql://root:@localhost:3306/dog_classifier_db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False 
    
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'a9b8c7d6e5f4g3h2i1j0192837465abcd1234567890abcdef')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(days=3) # <-- DISINI KADALUARSA TOKEN DIATUR KE 1 JAM
    # JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30) # Opsional: untuk refresh token yang lebih lama

    # --- Path untuk Model AI ---
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'dog_breed_classifier_final_model.keras')
    CLASS_INDICES_PATH = os.path.join(BASE_DIR, 'models', 'class_indices.json')

    # --- UKURAN GAMBAR ---
    # Ukuran gambar input yang diharapkan oleh model CNN
    IMG_HEIGHT = 224
    IMG_WIDTH = 224

    # --- Path untuk Folder Unggahan Gambar dan History Gambar ---
    # Folder sementara untuk menyimpan gambar yang baru diunggah oleh user sebelum diproses
    TEMP_UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'temp_uploads')
    # Folder permanen untuk menyimpan gambar history klasifikasi
    HISTORY_IMAGES_FOLDER = os.path.join(BASE_DIR, 'static', 'history_images')

    # --- Path untuk History Klasifikasi (jika masih menggunakan CSV paralel) ---
    # Jika Anda sepenuhnya beralih ke database, bagian ini bisa diabaikan atau dihapus
    HISTORY_FOLDER = os.path.join(BASE_DIR, 'data')
    HISTORY_FILE = os.path.join(HISTORY_FOLDER, 'classification_history.csv')