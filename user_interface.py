#!/usr/bin/python3

# app/user_interface.py
import cv2

from app.face_detection import FaceDetector
from app.face_alignment import FaceAligner
from app.feature_extraction import FeatureExtractor
from app.face_recognition import FacialRecognizer


def load_image(image_path):
    """
    Loads an image from the given path

    Args:
        image_path(str): The path to the image

    Returns:
        numpy.ndarray: The loaded image as a Numpy array
    """

class UserInterface(FacialRecognizer):
    """
    Class for the user interface functionality.
    """

    def __init__(self, shape_predictor_path, database_manager):
        """
        Initializes the UserInterface.

        Args:
            database_manager (DatabaseManager): An instance of the DatabaseManager class.
        """
        """self.database_manager = database_manager"""
        database = database_manager.get_database()
        super().__init__(shape_predictor_path, database)

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


    def display_results(self, recognized_faces):
        """
        Displays the recognized face IDs.

        Args:
            recognized_faces (list): A list of recognized face IDs.
        """
        for name in recognized_faces:
            print(f"Recognized face: {name}")


    def recognize_faces(self, image_path):
        """

        Recognizes faces in an image.

        Args:
            image_path (str): The path to the image file.

        Returns:
            list: A list of recognized face IDs.
        """
        """
        # Load the image
        image = load_image(image_path)

        # Detect faces in the image
        cascade_path = "data/haarcascade_frontalface_default.xml"
        face_detector = FaceDetector(cascade_path)
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
        """
        image = cv2.imread(image_path)
        return super().recognize_faces(image)
