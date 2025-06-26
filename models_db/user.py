# my_dog_breed_classifier/models_db/user.py

# Import db secara absolut dari extensions.py
from extensions import db 
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import create_access_token # Untuk membuat JWT dari user
from datetime import datetime # Untuk kolom created_at

class User(db.Model):
    __tablename__ = 'users' # Nama tabel di database

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False) # Tingkatkan ukuran untuk hash password
    name = db.Column(db.String(120), nullable=False) # Kolom 'name' baru
    created_at = db.Column(db.DateTime, default=datetime.utcnow) # Waktu pembuatan user

    # Relationship to PredictionHistory (jika history juga disimpan di DB)
    # predictions akan berisi daftar objek PredictionHistory yang terkait dengan user ini
    # backref='user' membuat properti 'user' di objek PredictionHistory yang merujuk kembali ke User.
    predictions = db.relationship('PredictionHistory', backref='user', lazy=True)

    def set_password(self, password):
        """Menghash password dan menyimpannya."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Memverifikasi password yang diberikan dengan hash yang tersimpan."""
        return check_password_hash(self.password_hash, password)

    def get_jwt_token(self):
        """Membuat JWT token untuk user ini."""
        # Token akan berisi user_id sebagai identity
        return create_access_token(identity=self.id)

    def __repr__(self):
        return f'<User {self.username}>'