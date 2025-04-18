from flask import Flask, render_template, request, jsonify, Response
import os
import cv2
import numpy as np
import json
import time
from datetime import datetime
from werkzeug.utils import secure_filename
import tweepy
import pandas as pd
from models.text_models import TextEmotionAnalyzer
from models.vision_models import ImageEmotionAnalyzer, VideoEmotionAnalyzer
from utils.twitter_utils import TwitterClient
import nltk

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)

# Initialize models
text_analyzer = TextEmotionAnalyzer()
image_analyzer = ImageEmotionAnalyzer()
video_analyzer = VideoEmotionAnalyzer()
twitter_client = TwitterClient()

# Global variable for video stream state
video_stream_active = False
video_camera = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
@app.route('/text')
def text_analysis():
    models = text_analyzer.get_available_models()
    return render_template('text_analysis.html', models=models, active_tab='text')

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    text = request.form.get('text')
    model_name = request.form.get('model')
    
    if not text:
        return jsonify({'error': 'No text provided'}),200
    
    try:
        results = text_analyzer.analyze(text, model_name)
        return jsonify(results),200
    except Exception as e:
        return jsonify({'error': str(e)}),200

@app.route('/image')
def image_analysis():
    models = image_analyzer.get_available_models()
    return render_template('image_analysis.html', models=models, active_tab='image')

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}),200
    
    file = request.files['image']
    model_name = request.form.get('model')
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}),200
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            results = image_analyzer.analyze(filepath, model_name)
            results['image_path'] = filepath.replace('static/', '')  # For displaying in template
            return jsonify(results),200
        except Exception as e:
            return jsonify({'error': str(e)}),200
    
    return jsonify({'error': 'Invalid file format'})

@app.route('/video')
def video_analysis():
    models = video_analyzer.get_available_models()
    return render_template('video_analysis.html', models=models, active_tab='video')

def generate_frames():
    global video_camera, video_stream_active
    
    # Initialize video capture
    video_camera = cv2.VideoCapture(0)
    
    while video_stream_active:
        success, frame = video_camera.read()
        if not success:
            break
        
        # Process frame with emotion detection
        processed_frame, emotion_results = video_analyzer.analyze_frame(frame)
        
        # Convert to jpeg format
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in byte format with emotion data
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
               b'Emotion-Data: ' + json.dumps(emotion_results).encode() + b'\r\n\r\n')
    
    # Release resources
    if video_camera:
        video_camera.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_video', methods=['POST'])
def start_video():
    global video_stream_active
    video_stream_active = True
    model_name = request.form.get('model', 'default')
    video_analyzer.set_model(model_name)
    return jsonify({'status': 'Video stream started'}),200

@app.route('/stop_video', methods=['POST'])
def stop_video():
    global video_stream_active, video_camera
    video_stream_active = False
    if video_camera:
        video_camera.release()
        video_camera = None
    return jsonify({'status': 'Video stream stopped'}),200

@app.route('/twitter')
def twitter_analysis():
    # Get available text models for Twitter analysis
    models = text_analyzer.get_available_models()
    return render_template('twitter_analysis.html', models=models, active_tab='twitter')

@app.route('/analyze_tweets', methods=['POST'])
def analyze_tweets():
    keyword = request.form.get('keyword')
    model_name = request.form.get('model')
    
    if not keyword:
        return jsonify({'error': 'No keyword provided'}),200
    
    try:
        # Get tweets
        tweets = twitter_client.search_tweets(keyword, count=10)
        
        # Analyze emotions in tweets
        results = []
        for tweet in tweets:
            emotion_results = text_analyzer.analyze(tweet.text, model_name)
            
            results.append({
                'username': tweet.user.screen_name,
                'timestamp': tweet.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'text': tweet.text,
                'emotions': emotion_results
            })
        
        return jsonify(results),200
    except Exception as e:
        return jsonify({'error': str(e)}),200

@app.route('/model_info/<model_name>')
def model_info(model_name):
    """Return information about a specific model"""
    model_info = {
        'TextBlob': {
            'description': 'TextBlob is a simple NLP library that provides sentiment analysis through a polarity score.',
            'output': 'Returns polarity scores mapped to positive, neutral, and negative categories.',
            'use_case': 'Best for simple sentiment analysis of short texts.'
        },
        'VADER': {
            'description': 'VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool.',
            'output': 'Returns negative, neutral, and positive scores, along with a compound score.',
            'use_case': 'Specialized for social media text and works well with informal language.'
        },
        'Flair': {
            'description': 'Flair is a powerful NLP library that provides state-of-the-art text classification.',
            'output': 'Returns binary sentiment classification (positive/negative) with confidence score.',
            'use_case': 'Good for more complex sentiment analysis with higher accuracy.'
        },
        'BERT Emotion': {
            'description': 'BERT is a transformer-based model pre-trained on a large corpus of text.',
            'output': 'Returns probabilities for 6 emotions: sadness, joy, love, anger, fear, and surprise.',
            'use_case': 'Best for detailed emotion analysis beyond basic sentiment.'
        },
        'RoBERTa Emotion': {
            'description': 'RoBERTa is an optimized version of BERT with improved training methodology.',
            'output': 'Returns probabilities for 7 emotions: anger, disgust, fear, joy, neutral, sadness, and surprise.',
            'use_case': 'Provides the most nuanced emotion analysis with state-of-the-art performance.'
        }
    }
    
    return jsonify(model_info.get(model_name, {'description': 'Model information not available.'})),200

if __name__ == '__main__':
    app.run(debug=True)