#!/usr/bin/python3

# app/database_manager.py
from sqlalchemy import create_engine, Column, Integer, String, PickleType
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Face(Base):
    """
    Class representing the Face entity in the database.
    """
    __tablename__ = 'faces'

    id = Column(Integer, primary_key=True)
    face_id = Column(String(50), unique=True)
    face_image = Column(PickleType)
    info = Column(PickleType)

    def __init__(self, face_id, face_image, info=None):
        self.face_id = face_id
        self.face_image = face_image
        self.info = info

class DatabaseManager:
    """
    Class for managing the reference database of facial images using MySQL and SQLAlchemy.
    """

    def __init__(self, database_url):
        """
        Initializes the DatabaseManager.

        Args:
            database_url (str): The URL for connecting to the MySQL database.
        """
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def add_face(self, face_id, face_image, info=None):
        """
        Adds a face to the reference database.

        Args:
            face_id (str): The ID of the face to add.
            face_image (numpy.ndarray): The face image to add.
            info (dict, optional): Additional information about the face. Defaults to None.
        """
        face = Face(face_id=face_id, face_image=face_image, info=info)
        self.session.add(face)
        self.session.commit()

    def remove_face(self, face_id):
        """
        Removes a face from the reference database.

        Args:
            face_id (str): The ID of the face to remove.
        """
        face = self.session.query(Face).filter_by(face_id=face_id).first()
        if face:
            self.session.delete(face)
            self.session.commit()

    def update_face(self, face_id, new_info):
        """
        Updates the information of a face in the reference database.

        Args:
            face_id (str): The ID of the face to update.
            new_info (dict): The new information to update.
        """
        face = self.session.query(Face).filter_by(face_id=face_id).first()
        if face:
            face.info = new_info
            self.session.commit()

    def get_face(self, face_id):
        """
        Retrieves a face from the reference database.

        Args:
            face_id (str): The ID of the face to retrieve.

        Returns:
            tuple: The face image and information, if available. None otherwise.
        """
        face = self.session.query(Face).filter_by(face_id=face_id).first()
        if face:
            return face.face_image, face.info
        else:
            return None, None

    def get_all_faces(self):
        """
        Retrieves all faces from the reference database.

        Returns:
            dict: The reference database containing all faces.
        """
        all_faces = self.session.query(Face).all()
        database = {face.face_id: {'image': face.face_image, 'info': face.info} for face in all_faces}
        return database

