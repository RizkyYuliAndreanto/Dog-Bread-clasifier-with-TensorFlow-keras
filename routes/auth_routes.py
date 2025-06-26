# my_dog_breed_classifier/routes/auth_routes.py

from flask import Blueprint, request, jsonify # Mengimpor komponen dasar Flask
# create_access_token, jwt_required, get_jwt_identity adalah satu-satunya fungsi JWT yang dibutuhkan di sini sekarang
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity 

# Mengimpor instance db dari extensions.py (Import absolut yang benar)
from extensions import db

# Mengimpor model User dari models_db.user (Import absolut yang benar)
from models_db.user import User

# Mengimpor layanan autentikasi dari services.auth_service
# AuthService sekarang dipanggil untuk register/login
from services.auth_service import AuthService

# Membuat instance Blueprint untuk rute autentikasi
# Semua rute di Blueprint ini akan memiliki prefix '/api/auth'
auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

# --- Rute Registrasi Pengguna ---
@auth_bp.route('/register', methods=['POST'])
def register():
    """
    Rute untuk pendaftaran pengguna baru.
    Menerima 'username', 'password', dan 'name' dalam format JSON.
    """
    data = request.get_json() # Mendapatkan data JSON dari body permintaan
    username = data.get('username')
    password = data.get('password')
    name = data.get('name') # Ambil nama lengkap dari request

    # Validasi input dasar
    if not username or not password or not name:
        return jsonify({'message': 'Username, password, dan nama lengkap wajib diisi.'}), 400 # Bad Request
    
    # Panggil layanan autentikasi untuk mendaftarkan user
    # AuthService.register_user akan menangani hashing password dan penyimpanan ke DB
    result = AuthService.register_user(username, password, name) # Mengirim 'name' ke service
    
    if result['success']:
        return jsonify({'message': result['message']}), 201 # 201 Created
    else:
        # Jika registrasi gagal (misal: username sudah ada), kembalikan error yang sesuai
        return jsonify({'error': result['message']}), 409 # 409 Conflict (jika username sudah ada)

# --- Rute Login Pengguna ---
@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Rute untuk login pengguna.
    Menerima 'username' dan 'password' dalam format JSON.
    """
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    # Validasi input dasar
    if not username or not password:
        return jsonify({'message': 'Username dan password wajib diisi.'}), 400 # Bad Request

    # Panggil layanan autentikasi untuk login user
    # AuthService.login_user akan memverifikasi password dan menghasilkan JWT
    result = AuthService.login_user(username, password)
    
    if result['success']:
        # Jika login berhasil, kembalikan pesan, token akses, dan user_id ke frontend
        return jsonify({
            'message': result['message'],
            'access_token': result['access_token'],
            'user_id': result['user_id']
        }), 200 # OK
    else:
        # Jika login gagal (misal: username/password salah), kembalikan error 401
        return jsonify({'error': result['message']}), 401 # 401 Unauthorized

# --- Rute Contoh Terlindungi (Opsional, untuk Pengujian) ---
@auth_bp.route('/protected', methods=['GET'])
@jwt_required() # <-- Ini membuat rute ini hanya bisa diakses dengan JWT yang valid
def protected():
    """
    Rute contoh yang hanya dapat diakses oleh pengguna yang terautentikasi.
    Menunjukkan cara mendapatkan identitas pengguna dari token.
    """
    # get_jwt_identity() akan mengembalikan nilai yang Anda set sebagai 'identity' saat membuat token (yaitu user.id)
    current_user_id = get_jwt_identity() 
    user = User.query.get(current_user_id) # Ambil objek User dari database berdasarkan ID

    if user:
        return jsonify(
            logged_in_as=user.username,
            name=user.name, # Mengirimkan nama lengkap
            message="Anda memiliki akses ke rute terlindungi!"
        ), 200
    return jsonify({"message": "User tidak ditemukan"}), 404 # Seharusnya tidak terjadi jika token valid
