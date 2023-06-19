#!/usr/bin/python3

import cv2
import os
import secrets
import uuid

from app.face_alignment import FaceAligner
from app.face_detection import FaceDetector
from app.face_encoding import FaceEncoder
from app.face_recognition import FacialRecognizer
from app.feature_extraction import FeatureExtractor
from app.database_operations import DatabaseManager
from flask import request, redirect, url_for, Flask, render_template, flash
from werkzeug.utils import secure_filename


shape_predictor_path = "data/shape_predictor_68_face_landmarks.dat"
database_url = "mysql+pymysql://facex:Face_x@localhost/face_recognition"
secret_key = secrets.token_hex(16)

app = Flask(__name__)

# Initialize the face recognition system
feature_extractor = FeatureExtractor(shape_predictor_path)
face_aligner = FaceAligner(shape_predictor_path)
face_encoder = FaceEncoder(feature_extractor)
database_manager = DatabaseManager(database_url)

training_dataset_folder = "images/training_dataset"
app.config['TRAINING_DATASET_FOLDER'] = training_dataset_folder
app.config.from_pyfile('config.py')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Check if a file was submitted
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('home'))

    file = request.files['file']

    # Check if the file is empty
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('home'))

    # Perform face recognition on the uploaded image
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Load the image using OpenCV
        image = cv2.imread(file_path)

        # Detect faces in the image
        cascade_path = "data/haarcascade_frontalface_default.xml"
        face_detector = FaceDetector(cascade_path)
        faces = face_detector.detect_faces(image)

        if len(faces) == 0:
            flash('No faces detected')
            return redirect(url_for('home'))

        # Align faces
        aligned_faces = []
        for face in faces:
            aligned_face = face_aligner.align_face(image, face)
            if aligned_face is not None:
                aligned_faces.append(aligned_face)

        if not aligned_faces:
            flash('No faces aligned')
            return redirect(url_for('home'))

        # Encode faces
        face_encodings = []
        for aligned_face in aligned_faces:
            encoding = face_encoder.encode_face(aligned_face)
            face_encodings.append(encoding)

        # Recognize faces
        recognized_faces = FacialRecognizer.recognize_faces(image, face_encodings)

        if not recognized_faces:
            flash('No faces recognized')
            return redirect(url_for('home'))

        # Update face recognition model or database with the training data
        for encoding in face_encodings:
            # Update your face recognition model or database with the encoding
            database_manager.add_face(face_id=generate_unique_id(), face_image=encoding)


        return render_template('results.html', recognized_faces=recognized_faces)

    flash('Invalid file type')
    return redirect(url_for('home'))

...
@app.route('/upload_training', methods=['GET'])
def upload_training():
    return render_template('upload.html')

@app.route('/process_training', methods=['POST'])
def process_training():
    # Check if files were submitted
    if 'files[]' not in request.files:
        flash('No files part')
        return redirect(url_for('upload_training'))

    files = request.files.getlist('files[]')

    # Process each file
    for file in files:
        # Check if the file is empty
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('upload_training'))

        # Save the file to the training dataset directory
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['TRAINING_DATASET_FOLDER'], filename)
        file.save(file_path)
        
        # Load the image using OpenCV
        image = cv2.imread(file_path)

        # Detect faces in the image
        cascade_path = "data/haarcascade_frontalface_default.xml"
        face_detector = FaceDetector(cascade_path)
        faces = face_detector.detect_faces(image)

        if len(faces) == 0:
            flash('No faces detected')
            return redirect(url_for('home'))

        
        # Perform face alignment
        aligned_faces = []
        for face in faces:
            aligned_face = face_aligner.align_face(image, face)
            if aligned_face is not None:
                aligned_faces.append(aligned_face)

        if not aligned_faces:
            flash('No faces aligned')
            return redirect(url_for('upload_training'))

        # Encode faces
        face_encodings = []
        for aligned_face in aligned_faces:
            encoding = face_encoder.encode_face(aligned_face)
            face_encodings.append(encoding)

        # Update face recognition model or database with the training data
        for encoding in face_encodings:
            # Update your face recognition model or database with the encoding
            database_manager.add_face(face_id=generate_unique_id(), face_image=encoding)

        # Perform any additional processing, such as face alignment, encoding, etc.
        # Update your face recognition model or database with the training data

    flash('Training dataset uploaded successfully')
    return redirect(url_for('home'))
...

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_unique_id():
    """
    Generate a unique ID for each face

    Return: 
        str: Unique ID
    """
    return str(uuid.uuid4())

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'images/uploads'
    app.secret_key = secret_key
    app.run(debug=True)

