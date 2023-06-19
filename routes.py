#!/usr/bin/python3

# app/routes.py
import os
from app.database_operations import DatabaseManager
from user_interface import UserInterface
from flask import Blueprint, jsonify
from flask import Flask, render_template, request, redirect

app = Flask(__name__)
main_routes = Blueprint('main', __name__)
shape_predictor_path = "data/shape_predictor_68_face_landmarks.dat"
database_url = "mysql+pymysql://facex:Face_x@localhost/face_recognition"


# Create an instance of the UserInterface
database_manager = DatabaseManager(database_url)
user_interface = UserInterface(shape_predictor_path, database_manager)

app.config.from_pyfile('config.py')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400

    image_file = request.files['image']
    recognized_faces = user_interface.recognize_faces(image_file)

    rendered_template = render_template('results.html', recognized_faces=recognized_faces)
    # json_response = jsonify({'recognized_faces': recognized_faces})

    return rendered_template, 200, {'Content-Type': 'text/html'}, #json_response


if __name__ == '__main__':
    app.run()

