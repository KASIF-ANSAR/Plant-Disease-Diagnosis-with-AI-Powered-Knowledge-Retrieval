# models.py
from db import db
from datetime import datetime

# ---------------- USER TABLE ----------------
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    uploads = db.relationship('Upload', backref='user', lazy=True)
    questions = db.relationship('Question', backref='user', lazy=True)

# ---------------- UPLOAD TABLE ----------------
class Upload(db.Model):
    __tablename__ = 'uploads'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    questions = db.relationship('Question', backref='upload', lazy=True)

# ---------------- QUESTION TABLE ----------------
class Question(db.Model):
    __tablename__ = 'questions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    upload_id = db.Column(db.Integer, db.ForeignKey('uploads.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    summary = db.Column(db.String(255))   
    answer = db.Column(db.Text, nullable=False)
    asked_at = db.Column(db.DateTime, default=datetime.utcnow)
