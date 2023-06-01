#!/usr/bin/python3

import numpy as np

class FaceEncoder:
    """
    Class for encoding faces using the FeatureExtractor.
    """

    def __init__(self, feature_extractor):
        """
        Initializes the FaceEncoder with the provided FeatureExtractor.

        Args:
            feature_extractor (FeatureExtractor): The FeatureExtractor instance.
        """
        self.feature_extractor = feature_extractor

    def encode_face(self, aligned_face):
        """
        Encodes the given aligned face using the FeatureExtractor.

        Args:
            aligned_face (numpy.ndarray): The aligned face image.

        Returns:
            numpy.ndarray: The encoded face.
        """
        features = self.feature_extractor.extract_features(aligned_face)
        return features

"""
# Example usage
aligned_face = ...  # Aligned face image

# Initialize the FeatureExtractor (assuming you already have it)
feature_extractor = FeatureExtractor()

# Initialize the FaceEncoder with the FeatureExtractor
face_encoder = FaceEncoder(feature_extractor)

# Encode the face
encoded_face = face_encoder.encode_face(aligned_face)

# Use the encoded face for further processing or comparison
"""
