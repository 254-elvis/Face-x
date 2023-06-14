#!/usr/bin/python3

# app/user_interface.py
from app.face_detection import FaceDetector
from app.face_alignment import FaceAligner
from app.feature_extraction import FeatureExtractor
from app.face_recognition import FacialRecognizer


class UserInterface:
    """
    Class for the user interface functionality.
    """

    def __init__(self, database_manager):
        """
        Initializes the UserInterface.

        Args:
            database_manager (DatabaseManager): An instance of the DatabaseManager class.
        """
        self.database_manager = database_manager

    def add_face(self, face_id, face_image_path, info=None):
        """
        Adds a face to the reference database.

        Args:
            face_id (str): The ID of the face to add.
            face_image_path (str): The path to the face image file.
            info (dict, optional): Additional information about the face. Defaults to None.
        """
        # Load the face image from the file
        face_image = load_image(face_image_path)

        # Add the face to the database
        self.database_manager.add_face(face_id, face_image, info)

    def remove_face(self, face_id):
        """
        Removes a face from the reference database.

        Args:
            face_id (str): The ID of the face to remove.
        """
        self.database_manager.remove_face(face_id)

    def update_face(self, face_id, new_info):
        """
        Updates the information of a face in the reference database.

        Args:
            face_id (str): The ID of the face to update.
            new_info (dict): The new information to update.
        """
        self.database_manager.update_face(face_id, new_info)

    def recognize_faces(self, image_path):
        """
        Recognizes faces in an image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            list: A list of recognized face IDs.
        """
        # Load the image
        image = load_image(image_path)

        # Detect faces in the image
        face_detector = FaceDetector()
        detected_faces = face_detector.detect_faces(image)

        # Align and extract features from the detected faces
        face_aligner = FaceAligner()
        feature_extractor = FeatureExtractor()
        aligned_faces = []
        for face in detected_faces:
            aligned_face = face_aligner.align_face(face)
            features = feature_extractor.extract_features(aligned_face)
            aligned_faces.append((aligned_face, features))

        # Get the reference database
        database = self.database_manager.get_all_faces()

        # Perform face recognition
        face_recognizer = FaceRecognizer()
        recognized_faces = []
        for aligned_face, features in aligned_faces:
            face_id = face_recognizer.match_faces(features, database)
            recognized_faces.append(face_id)

        return recognized_faces

    def display_results(self, recognized_faces):
        """
        Displays the recognized face IDs.

        Args:
            recognized_faces (list): A list of recognized face IDs.
        """
        for face_id in recognized_faces:
            print(f"Recognized face: {face_id}")

