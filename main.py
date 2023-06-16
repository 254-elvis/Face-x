#!/usr/bin/python3

from app.face_alignment import FaceAligner
from app.face_encoding import FaceEncoder
from app.feature_extraction import FeatureExtractor
from app.database_operations import DatabaseManager
from flask import request, redirect, url_for, Flask, render_template

shape_predictor_path = "data/shape_predictor_68_face_landmarks.dat"
database_url = "mysql+pymysql://facex:Face_x@localhost/face_recognition"

app = Flask(__name__)

# Initialize the face recognition system
feature_extractor = FeatureExtractor(shape_predictor_path)
face_aligner = FaceAligner(shape_predictor_path)
face_encoder = FaceEncoder(feature_extractor)
database_manager = DatabaseManager(database_url)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Check if a file was submitted
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        return redirect(url_for('home'))

    # Perform face recognition on the uploaded image
    result = recognize_faces(file)

    return render_template('result.html', result=result)

def recognize_faces(image):
    # Preprocess the image
    aligned_faces = face_aligner.align_face(image, face)

    # Encode the faces
    face_encodings = []
    for aligned_face in aligned_faces:
        face_encoding = face_encoder.encode_face(aligned_face)
        face_encodings.append(face_encoding)

    # Perform face recognition using the encoded faces
    recognized_faces = database_manager.recognize_faces(face_encodings)

    return recognized_faces

if __name__ == '__main__':
    app.run(debug=True)

