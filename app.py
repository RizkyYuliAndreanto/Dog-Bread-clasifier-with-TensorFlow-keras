# my_dog_breed_classifier/app.py

from flask import Flask, jsonify
import os
from flask_cors import CORS # Import CORS (sudah di extensions, tapi ini untuk inisialisasi di sini)

from config import Config # Import konfigurasi dari config.py
from routes import register_blueprints # Import fungsi untuk mendaftarkan Blueprints
# Import instance db, jwt, cors, migrate dari extensions.py
from extensions import db, jwt, cors, migrate 

def create_app():
    """
    Fungsi pabrik untuk membuat dan mengonfigurasi instance aplikasi Flask.
    Ini adalah praktik terbaik untuk aplikasi Flask yang lebih besar dan modular.
    """
    # Inisialisasi Flask app.
    # 'static_folder='static'' memberitahu Flask untuk menyajikan file dari folder 'static'.
    # 'instance_relative_config=True' memungkinkan pemuatan konfigurasi dari folder 'instance' (jika ada file config di sana).
    app = Flask(__name__, static_folder='static', instance_relative_config=True)
    
    # Muat konfigurasi dari objek Config yang kita definisikan.
    app.config.from_object(Config)

    # --- Inisialisasi Ekstensi dengan Aplikasi Flask ---
    # Inisialisasi instance database SQLAlchemy dengan aplikasi Flask.
    db.init_app(app) 
    # Inisialisasi JWT Manager dengan aplikasi Flask.
    jwt.init_app(app)
    
    # Inisialisasi CORS dengan aplikasi Flask.
    # Tentukan origins secara eksplisit saat supports_credentials=True
    # http://localhost:5173 adalah default development server untuk Vue 3 (Vite)
    cors.init_app(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"], supports_credentials=True, allow_headers=["Authorization", "Content-Type"])

    # Inisialisasi Flask-Migrate dengan aplikasi Flask dan instance database.
    migrate.init_app(app, db) 

    # --- JWT CALLBACKS GLOBAL ---
    # Callbacks ini harus didaftarkan pada instance 'jwt' setelah 'jwt.init_app(app)'
    # untuk menangani respons error JWT secara global di aplikasi.
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

    # --- Pastikan Folder-folder yang Dibutuhkan Ada ---
    # Fungsi-fungsi ini akan membuat folder jika belum ada di path yang ditentukan.
    if not os.path.exists(app.config['TEMP_UPLOAD_FOLDER']):
        os.makedirs(app.config['TEMP_UPLOAD_FOLDER'])
    if not os.path.exists(app.config['HISTORY_IMAGES_FOLDER']):
        os.makedirs(app.config['HISTORY_IMAGES_FOLDER'])
    if not os.path.exists(app.config['HISTORY_FOLDER']):
        os.makedirs(app.config['HISTORY_FOLDER'])
    
    # --- Membuat Tabel Database ---
    # db.create_all() harus dipanggil setelah db.init_app(app) dan di dalam app_context().
    with app.app_context():
        # Impor model-model database di sini agar dikenal oleh db.create_all()
        # Ini penting agar SQLAlchemy dapat menemukan dan membuat tabelnya di database.
        from models_db.user import User 
        from models_db.prediction_history import PredictionHistory 
        db.create_all() # Membuat tabel jika belum ada

    # --- Daftarkan Semua Blueprints ---
    # Mendaftarkan semua kelompok rute (main, auth, api) ke aplikasi Flask.
    register_blueprints(app)

    # --- HAPUS ROUTE ROOT INI AGAR BLUEPRINT main_bp DAPAT MENGAMBIL ALIH ---
    # @app.route('/')
    # def index():
    #     return "Dog Breed Classifier Backend is Running!"

    return app

# --- Jalankan Aplikasi Flask ---
# Bagian ini akan dieksekusi hanya jika 'app.py' dijalankan langsung
# (misalnya, dengan 'python app.py' atau 'flask run').
if __name__ == '__main__':
    app = create_app()
    # Menjalankan aplikasi dengan konfigurasi debug, host, dan port dari Config
    app.run(debug=app.config['DEBUG'], host=app.config['HOST'], port=app.config['PORT'])