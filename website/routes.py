from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from .models import User, Thread, Post
from . import db
from flask_login import login_required, current_user

routes = Blueprint('routes', __name__)

@routes.route('/')
def index():
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login'))
    return redirect(url_for('routes.home'))

@routes.route('/home')
@login_required
def home():
    return render_template('home.html')

@routes.route('/forum')
@login_required
def forum():
    return render_template('forum.html')

@routes.route('/create_thread', methods=['POST'])
@login_required
def create_thread():
    title = request.form['title']
    new_thread = Thread(title=title, author_id=current_user.id)
    db.session.add(new_thread)
    db.session.commit()
    return redirect(url_for('routes.home'))

@routes.route('/thread/<int:thread_id>', methods=['GET', 'POST'])
@login_required
def thread(thread_id):
    thread = Thread.query.get_or_404(thread_id)
    posts = Post.query.filter_by(thread_id=thread_id).all()

    if request.method == 'POST':
        content = request.form.get('content')
        new_post = Post(content=content, author_id=current_user.id, thread_id=thread_id)
        db.session.add(new_post)
        db.session.commit()
        flash('Post added!', category='success')
        return redirect(url_for('routes.thread', thread_id=thread_id))

    is_author = (thread.author_id == current_user.id)
    return render_template('thread.html', thread=thread, posts=posts, is_author=is_author)

@routes.route('/delete_thread/<int:thread_id>', methods=['POST'])
@login_required
def delete_thread(thread_id):
    thread = Thread.query.get(thread_id)
    if thread and thread.author_id == current_user.id:
        db.session.delete(thread)
        db.session.commit()
        flash('Thread deleted successfully!', category='success')
    else:
        flash('You are not authorized to delete this thread.', category='error')
    return redirect(url_for('routes.home'))

@routes.route('/thread/<int:thread_id>/add_post', methods=['POST'])
@login_required
def add_post(thread_id):
    thread = Thread.query.get_or_404(thread_id)
    content = request.form.get('content')
    if content:
        new_post = Post(content=content, author_id=current_user.id, thread_id=thread.id)
        db.session.add(new_post)
        db.session.commit()
        flash('Post added successfully!', category='success')
    return redirect(url_for('routes.thread', thread_id=thread_id))

import torch
import os
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet

IMAGE_PATH = "website\\static\\images\\eye1.jpeg"
CHECKPOINT_PATH = "C:\\Users\\Saite\\OneDrive\\Coding Folder\\Hackathon\\SIH_2024\\website\\b3_2.pth.tar"

def load_checkpoint(checkpoint, model, optimizer=None, lr=None):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    
    if optimizer is not None and lr is not None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

def load_model(checkpoint_path, device):
    # Define the model architecture
    model = EfficientNet.from_name("efficientnet-b3")
    model._fc = torch.nn.Linear(1536, 5)  # Adjust output layer for 5 classes
    model = model.to(device)

    # Load the model checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        load_checkpoint(checkpoint, model)  # No need for optimizer or lr here
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_image(image_path):
    # Define the transformations
    transform = A.Compose([
        A.Resize(height=120, width=120),
        A.Normalize(
            mean=[0.3199, 0.2240, 0.1609],
            std=[0.3020, 0.2183, 0.1741],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image=np.array(image))  # Convert PIL image to numpy array for albumentations
    image = image['image'].unsqueeze(0)  # Add batch dimension
    return image

def make_prediction(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        prediction = outputs.argmax(dim=1)
    return prediction.item()

@routes.route('/predict', methods=['POST', 'GET'])
@login_required
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:  
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']  
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            try:
                image_filename = file.filename
                image_path = os.path.join('website', 'static', 'uploads', file.filename)
                file.save(image_path)
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = load_model(CHECKPOINT_PATH, device)
                
                image_tensor = preprocess_image(image_path)
                
                prediction = make_prediction(model, image_tensor, device)
                
                image_url = url_for('static', filename=f'uploads/{image_filename}')
                
                return render_template('eyediseasepred.html', prediction=f"Predicted class: {prediction}", image_url=image_url)
            except Exception as e:
                flash(f'Error: {str(e)}')
                return redirect(request.url)

    return render_template('eyediseasepred.html', prediction="Error in processing image.")

