# my_dog_breed_classifier/routes/__init__.py

from flask import Blueprint

# Impor Blueprint dari file terpisah
from .main_routes import main_bp
from .auth_routes import auth_bp # Import Blueprint Autentikasi
from .api_routes import api_bp  # Import Blueprint API Klasifikasi

def register_blueprints(app):
    """Mendaftarkan semua Blueprints ke aplikasi Flask."""
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp) # Daftarkan Blueprint Autentikasi
    app.register_blueprint(api_bp)  # Daftarkan Blueprint API Klasifikasi