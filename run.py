# my_dog_breed_classifier/run.py

# File ini adalah titik masuk untuk perintah 'flask' CLI.
# FLASK_APP yang diatur ke 'run.py' akan menemukan fungsi 'create_app()'.

from app import create_app # Import fungsi create_app dari app.py
from extensions import db # Import instance 'db' dari extensions.py
from models_db.user import User # Import model-model Anda untuk shell context
from models_db.prediction_history import PredictionHistory # Import model-model Anda untuk shell context

app = create_app()

# Menambahkan User dan PredictionHistory ke shell context
# Ini berguna jika Anda ingin berinteraksi dengan database dari Flask shell (flask shell)
@app.shell_context_processor
def make_shell_context():
    return dict(app=app, db=db, User=User, PredictionHistory=PredictionHistory)

# Anda tidak perlu baris 'if __name__ == '__main__': app.run(...)' di sini
# karena Anda akan menggunakan perintah 'flask run' dari terminal.