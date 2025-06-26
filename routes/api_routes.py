# my_dog_breed_classifier/routes/api_routes.py

from flask import Blueprint, request, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np # Pastikan ini diimpor
import pandas as pd # Pastikan ini diimpor
from datetime import datetime
import shutil # Untuk operasi salin/hapus file

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
@jwt_required(optional=True) # <-- PERUBAHAN DI SINI: Prediksi dapat dilakukan oleh guest atau logged-in user
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
        # --- Simpan File yang Diunggah Sementara ---
        # Mengamankan nama file untuk mencegah Path Traversal
        filename = secure_filename(file.filename)
        # Path lengkap untuk file sementara di folder temp_uploads
        temp_filepath = os.path.join(Config.TEMP_UPLOAD_FOLDER, filename)
        file.save(temp_filepath)

        # --- Pra-pemrosesan Gambar dan Prediksi ---
        processed_image = preprocess_image(temp_filepath)
        predictions = model.predict(processed_image)[0]
        
        predicted_class_index = np.argmax(predictions)
        confidence = float(predictions[predicted_class_index])
        predicted_breed = idx_to_class.get(predicted_class_index, 'Ras Tidak Dikenali')

        # --- Logika Penyimpanan Gambar History Permanen ---
        # Buat timestamp dan string acak untuk nama file unik
        timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_filename = f"{timestamp_str}_{os.urandom(4).hex()}_{filename}" 
        # Path lengkap untuk menyimpan gambar secara permanen di folder history_images
        history_image_storage_path = os.path.join(Config.HISTORY_IMAGES_FOLDER, unique_filename)
        
        # Salin file dari folder sementara ke folder history permanen
        shutil.copy(temp_filepath, history_image_storage_path)
        
        # Hapus file sementara setelah disalin
        os.remove(temp_filepath)

        # --- Catat Hasil ke History Database Menggunakan Service ---
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
        # --- Penanganan Error ---
        # Pastikan file sementara dihapus jika terjadi error selama pemprosesan
        if temp_filepath and os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        print(f"Error saat memproses permintaan prediksi: {e}")
        return jsonify({'error': f'Terjadi kesalahan saat memproses gambar: {str(e)}'}), 500 # Internal Server Error

# --- ROUTE UNTUK MENGAMBIL HISTORY KLASIFIKASI (SEKARANG PUBLIK) ---
@api_bp.route('/history', methods=['GET'])
# @jwt_required() # <-- PERUBAHAN DI SINI: DECORATOR INI DIHAPUS UNTUK AKSES PUBLIK PENUH
def get_history():
    """
    Rute untuk mengambil SEMUA history prediksi ras anjing (publik).
    Tidak memerlukan autentikasi JWT.
    """
    try:
        # Panggil get_all_history untuk mengambil semua data history
        history_list = ClassificationService.get_all_history() # <-- Panggil fungsi baru ini
        return jsonify(history_list), 200 # OK
    except Exception as e:
        print(f"Error saat mengambil history: {e}")
        return jsonify({'error': f'Gagal mengambil history: {str(e)}'}), 500 # Internal Server Error
