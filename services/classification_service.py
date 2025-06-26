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
        print(f"DEBUG: Mencoba menyimpan prediksi untuk {predicted_breed} (user_id: {user_id}, file: {image_filename})") # DEBUG LINE 1
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
            print("DEBUG: new_history ditambahkan ke sesi DB. Mencoba commit...") # DEBUG LINE 2
            db.session.commit()
            print(f"DEBUG: Commit berhasil untuk prediksi {predicted_breed}!") # DEBUG LINE 3
            return {'success': True, 'message': 'History berhasil disimpan.'}
        except Exception as e:
            db.session.rollback() # Rollback jika terjadi error
            print(f"ERROR: Terjadi kesalahan saat menyimpan history ke DB: {e}") # DEBUG LINE 4 (INI PENTING)
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

        print(f"DEBUG: Memanggil get_user_history() untuk user_id: {user_id}...") # DEBUG LINE 5
        try:
            # Query semua record history untuk user_id tertentu, diurutkan berdasarkan timestamp terbaru
            history_records = PredictionHistory.query.filter_by(user_id=user_id).order_by(PredictionHistory.timestamp.desc()).all()
            print(f"DEBUG: Ditemukan {len(history_records)} record history dari DB untuk user {user_id}.") # DEBUG LINE 6
            
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
            print(f"DEBUG: history_list siap untuk user {user_id}: {len(history_list)} item.") # DEBUG LINE 7
            return history_list
        except Exception as e:
            print(f"ERROR: Terjadi kesalahan saat mengambil history user {user_id} dari DB: {e}") # DEBUG LINE 8 (INI PENTING)
            return []

    @staticmethod
    def get_all_history():
        """
        Mengambil SEMUA history prediksi dari database (untuk akses publik).
        Mengembalikan list of dictionaries.
        """
        print("DEBUG: Memanggil get_all_history()...") # DEBUG LINE 9
        try:
            # Query semua record history, diurutkan berdasarkan timestamp terbaru
            history_records = PredictionHistory.query.order_by(PredictionHistory.timestamp.desc()).all()
            print(f"DEBUG: Ditemukan {len(history_records)} record history dari DB (semua).") # DEBUG LINE 10
            
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
            print(f"DEBUG: history_list (all) siap: {len(history_list)} item.") # DEBUG LINE 11
            return history_list
        except Exception as e:
            print(f"ERROR: Terjadi kesalahan saat mengambil semua history dari DB: {e}") # DEBUG LINE 12 (INI PENTING)
            return []

    # --- Opsional: Metode untuk History yang Disimpan di CSV (jika masih digunakan paralel) ---
    # Jika Anda sudah pindah sepenuhnya ke DB untuk history, bagian ini bisa dihapus.
    @staticmethod
    def save_prediction_history_to_csv(timestamp_str, unique_filename, relative_image_url, predicted_breed, confidence, user_id):
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
