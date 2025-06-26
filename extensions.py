# extensions.py
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_migrate import Migrate # <-- TAMBAHKAN INI

db = SQLAlchemy()
migrate = Migrate()
cors = CORS()
jwt = JWTManager()

def init_app(app):
    db.init_app(app)
    migrate.init_app(app, db)
    cors.init_app(app) # Inisialisasi CORS
    jwt.init_app(app)  # Inisialisasi JWT Manager