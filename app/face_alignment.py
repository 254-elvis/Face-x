#!/usr/bin/python3
"""
Class contains methods to align faces to standard postion
and orientation
"""

import cv2
import dlib


class FaceAligner:
    """
    Class for aligning faces to a standardized position and orientation.
    """

    def __init__(self, shape_predictor_path):
        """
        Initializes the FaceAligner with the provided shape predictor model path.

        Args:
            shape_predictor_path (str): Path to the shape predictor model file.
        """
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def align_face(self, image, face):
        """
        Aligns the detected face to a standardized position and orientation.

        Args:
            image (numpy.ndarray): The image containing the face.
            face (tuple): Bounding box coordinates of the detected face (x, y, w, h).

        Returns:
            numpy.ndarray: The aligned face image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (x, y, w, h) = face
        rect = dlib.rectangle(x, y, x + w, y + h)
        shape = self.shape_predictor(gray, rect)

        aligned_face = dlib.get_face_chip(image, shape)
        return aligned_face
"""
# Example usage
image = cv2.imread('image.jpg')
face = (x, y, w, h)  # Detected face coordinates

# Initialize the FaceAligner with the shape predictor model path
aligner = FaceAligner('shape_predictor_68_face_landmarks.dat')

# Align the detected face
aligned_face = aligner.align_face(image, face)

cv2.imshow('Aligned Face', aligned_face)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
