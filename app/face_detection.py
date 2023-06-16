#!/usr/bin/python3
"""
This is class containing methods to detect faces in a
photo or video
"""

import cv2


class FaceDetector:
    """
    Class for detecting faces in an image using OpenCV's 
    Haar cascade.
    """

    def __init__(self, cascade_path):
        """
        Initializes the FaceDetector with the given Haar cascade xml file
        """
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image):
        """
        Detects faces in the given image.

        Args:
            image (numpy.ndarray): The image to detect faces from

        Returns:
            List of tuples: Bounding box coordinates of the detected faces
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces
"""
# Usage
image = cv2.imread('image.jpg')

# Initialize the FaceDetector with the Haar cascade XML file path
detector = FaceDetector('haarcascade_frontalface_default.xml')

# Detect faces in the image
detected_faces = detector.detect_faces(image)

# Draw rectangles around the detected faces
for (x, y, w, h) in detected_faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

