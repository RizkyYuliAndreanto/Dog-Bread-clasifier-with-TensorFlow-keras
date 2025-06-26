# my_dog_breed_classifier/services/auth_service.py

from extensions import db # Mengimpor instance 'db' dari extensions.py
from models_db.user import User # Mengimpor model User dari models_db.user
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity

class AuthService:
    """
    Layanan yang menangani logika bisnis terkait autentikasi pengguna.
    """
    @staticmethod
    def register_user(username, password, name): # <-- PERUBAHAN DI SINI: Tambahkan 'name'
        """
        Mendaftarkan pengguna baru ke database.
        Mengembalikan dictionary {'success': bool, 'message': str}.
        """
        # Periksa apakah username sudah ada
        if User.query.filter_by(username=username).first():
            return {'success': False, 'message': 'Username sudah ada. Pilih yang lain.'}
        
        # Buat objek User baru dan set password (yang akan di-hash)
        # PERUBAHAN DI SINI: Inisialisasi User dengan 'name'
        new_user = User(username=username, name=name) 
        new_user.set_password(password)
        
        try:
            # Tambahkan user ke session database dan commit
            db.session.add(new_user)
            db.session.commit()
            return {'success': True, 'message': 'Registrasi berhasil. Silakan login.'}
        except Exception as e:
            # Rollback transaksi jika terjadi error
            db.session.rollback()
            print(f"Error saat registrasi user: {e}") # Log error ke konsol server
            return {'success': False, 'message': 'Gagal registrasi. Terjadi kesalahan server.'}

    @staticmethod
    def login_user(username, password):
        """
        Mengotentikasi pengguna.
        Mengembalikan dictionary {'success': bool, 'message': str, 'access_token': str, 'user_id': int}.
        """
        # Cari user berdasarkan username
        user = User.query.filter_by(username=username).first()
        
        # Periksa apakah user ada dan password cocok
        if user and user.check_password(password):
            # Jika login berhasil, buat JWT access token
            access_token = create_access_token(identity=user.id)
            return {
                'success': True,
                'message': 'Login berhasil.',
                'access_token': access_token,
                'user_id': user.id
            }
        else:
            return {'success': False, 'message': 'Username atau password salah.'}

    @staticmethod
    @jwt_required() # Rute ini memerlukan token JWT yang valid untuk diakses
    def get_current_user_id_from_token():
        """
        Mengambil ID pengguna dari token JWT yang aktif.
        Hanya bisa dipanggil di dalam rute yang dilindungi oleh @jwt_required().
        """
        return get_jwt_identity()
