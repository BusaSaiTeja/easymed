from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from .models import User, Thread, Post
from flask import jsonify
import requests
from . import db
from flask_login import login_required, current_user
import os
import torch
from .predict import load_checkpoint,load_model,make_prediction,preprocess_image,CHECKPOINT_PATH

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
    threads = Thread.query.all()  
    return render_template('forum.html', threads=threads)

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

@routes.route('/create_thread', methods=['POST'])
@login_required
def create_thread():
    title = request.form['title']
    new_thread = Thread(title=title, author_id=current_user.id)
    db.session.add(new_thread)
    db.session.commit()
    return redirect(url_for('routes.forum'))

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
    return redirect(url_for('routes.forum'))

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


@routes.route('/medicalbot', methods=['POST', 'GET'])
@login_required
def medicalbot():
    return render_template('medicalbot.html')

@routes.route('/map_view')
@login_required
def map_view():
    return render_template('maps.html')

@routes.route('/search')
@login_required
def search():
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    query_type = request.args.get('type')

    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      node["amenity"="{query_type}"](around:5000,{lat},{lon});
    );
    out body;
    """

    response = requests.get(overpass_url, params={'data': overpass_query})

    if response.status_code != 200:
        print('Error with Overpass API request:', response.status_code)
        return jsonify({'error': 'Failed to fetch data'}), 500

    data = response.json()

    results = []
    for element in data['elements']:
        if 'tags' in element and 'name' in element['tags']:
            results.append({
                'name': element['tags']['name'],
                'lat': element['lat'],
                'lon': element['lon']
            })

    return jsonify(results)
