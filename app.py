
import os
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, session
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads/"
MASK_FOLDER = "static/masks/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER 
app.config["MASK_FOLDER"] = MASK_FOLDER
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Add this for session management
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# Load trained model
model = load_model("C:/Users/tejas/OneDrive/Desktop/Breast_Cancer_Classifiction/Breast Cancer/Breast Cancer/breast_cancer_model.h5")

# Define classes
classes = ["Benign", "Malignant", "Normal"]

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Create all database tables
with app.app_context():
    db.create_all()

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def generate_mask(image_path, filename):
    """Generate a mask image using thresholding"""
    # Create full path for mask directory if it doesn't exist
    mask_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "masks")
    os.makedirs(mask_dir, exist_ok=True)
    
    # Read image in color
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image from {image_path}")
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to match the input size
    gray = cv2.resize(gray, (150, 150))
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Create a threshold mask
    _, mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Create full path for mask file
    mask_path = os.path.join(mask_dir, f"mask_{filename}")
    
    # Save the mask image
    success = cv2.imwrite(mask_path, mask)
    if not success:
        print(f"Failed to save mask to {mask_path}")
        return None
    
    print(f"Mask saved successfully to {mask_path}")  # Debug print
    return f"masks/mask_{filename}"  # Return relative path for URL

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['username'] = username
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

# Protect routes that require login
def login_required(route_function):
    def wrapper(*args, **kwargs):
        if 'username' not in session:
            flash('Please login first', 'error')
            return redirect(url_for('login'))
        return route_function(*args, **kwargs)
    wrapper.__name__ = route_function.__name__
    return wrapper

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            # Ensure filename is secure
            filename = file.filename
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Generate mask before preprocessing
            mask_path = generate_mask(filepath, filename)
            if mask_path is None:
                return "Error processing image", 400

            # Preprocess and predict
            img_array = preprocess_image(filepath)
            predictions = model.predict(img_array)
            predicted_class = classes[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

            return render_template("index.html", 
                                uploaded_image=url_for("static", filename=f"uploads/{filename}"),
                                mask_image=url_for("static", filename=f"masks/mask_{filename}"),
                                prediction=predicted_class, 
                                confidence=confidence)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
