#!/usr/bin/python3

import cv2
import dlib
import numpy as np


class FeatureExtractor:
    """
    Class for extracting facial features from aligned face images.
    """

    def __init__(self, shape_predictor_path):
        """
        Initializes the FeatureExtractor with the provided shape predictor model path.

        Args:
            shape_predictor_path (str): Path to the shape predictor model file.
        """
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def extract_features(self, aligned_face):
        """
        Extracts facial features from the aligned face image.

        Args:
            aligned_face (numpy.ndarray): The aligned face image.

        Returns:
            numpy.ndarray: The extracted facial features.
        """
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)
        shape = self.shape_predictor(gray, dlib.rectangle(0, 0, aligned_face.shape[0], aligned_face.shape[1]))
        landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(shape.num_parts)], dtype=np.float32)
        return landmarks

"""
# Example usage
aligned_face = ...  # Aligned face image

# Initialize the FeatureExtractor with the shape predictor model path
shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'
feature_extractor = FeatureExtractor(shape_predictor_path)

# Extract facial features
features = feature_extractor.extract_features(aligned_face)

# Use the extracted features for further processing or analysis
"""
