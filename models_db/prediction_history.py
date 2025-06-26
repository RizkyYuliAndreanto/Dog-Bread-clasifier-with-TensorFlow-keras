# my_dog_breed_classifier/models_db/prediction_history.py

# Import db secara absolut dari extensions.py
from extensions import db
from datetime import datetime # Untuk default timestamp

class PredictionHistory(db.Model):
    __tablename__ = 'prediction_history' # Nama tabel di database

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False) # Foreign Key ke tabel users
    timestamp = db.Column(db.DateTime, nullable=False, default=db.func.now()) # Waktu prediksi
    image_filename = db.Column(db.String(255), nullable=False) # Nama file gambar di folder static/history_images
    image_url = db.Column(db.String(255), nullable=False) # URL relatif gambar untuk frontend
    predicted_breed = db.Column(db.String(80), nullable=False)
    confidence = db.Column(db.Float, nullable=False) # Kepercayaan prediksi (sebagai float)

    def __repr__(self):
        return f'<Prediction {self.predicted_breed} by User {self.user_id}>'