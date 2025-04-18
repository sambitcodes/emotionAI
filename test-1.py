import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN
from fer import FER
import uuid
import os
from deepface import DeepFace

class BaseVisionModel:
    def __init__(self):
        # Initialize face detector
        self.face_detector = MTCNN(keep_all=True, device='cpu')
        
    def detect_faces(self, image):
        """Detect faces in image using MTCNN"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get face bounding boxes
        boxes, _ = self.face_detector.detect(image)
        
        if boxes is None:
            return []
        
        # Convert boxes to integer coordinates
        boxes = boxes.astype(int)
        
        # Return list of face regions
        face_regions = []
        for box in boxes:
            x1, y1, x2, y2 = box
            # Ensure coordinates are within image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            face_regions.append((x1, y1, x2, y2))
        
        return face_regions


class ImageEmotionAnalyzer(BaseVisionModel):
    def __init__(self):
        super().__init__()
        
        # Initialize models
        self.models = {
            'FER-2013': 'fer2013',
            'FER+': 'ferplus',
            'HSE': 'hse', # Human Sentiment Estimation  
            'DeepFace': 'deepface' # DeepFace model
        }
        
        # Default to FER
        self.emotion_detector = FER(mtcnn=False)  # We already have MTCNN
        self.current_model = 'FER-2013'
    
    def get_available_models(self):
        """Return list of available models"""
        return list(self.models.keys())
    
    def set_model(self, model_name):
        """Change emotion detection model"""
        if model_name in self.models:
            self.current_model = model_name
            # In a real implementation, we would actually load different models here
            # For simplicity, we're using FER for all models in this example
    
    def analyze(self, image_path, model_name=None):
        """Analyze emotions in an image"""
        if model_name:
            self.set_model(model_name)
        
        # Read image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_regions = self.detect_faces(image_rgb)
        
        # Create copy for visualization
        output_image = image.copy()
        
        # Analyze each face
        face_emotions = []


        #### DeepFace Analysis ####

        if self.current_model == 'DeepFace':
            try:
                analysis = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)

                # Normalize result to list
                if isinstance(analysis, dict):
                    analysis = [analysis]

                for result in analysis:
                    dominant_emotion = result['dominant_emotion']
                    emotion_scores = result['emotion']

                    face_emotions.append({
                        'primary_emotion': dominant_emotion,
                        'confidence': float(emotion_scores[dominant_emotion]) / 100.0,
                        'all_emotions': {k: float(v) / 100.0 for k, v in emotion_scores.items()}
                    })
            except Exception as e:
                print(f"DeepFace failed: {e}")

        else:
            face_regions = self.detect_faces(image_rgb)

            if len(face_regions) == 0:
                emotions = self.emotion_detector.detect_emotions(image_rgb)

                if len(emotions) > 0:
                    for emotion_data in emotions:
                        box = emotion_data['box']
                        x, y, w, h = box
                        emotions_dict = emotion_data['emotions']
                        primary_emotion = max(emotions_dict.items(), key=lambda x: x[1])

                        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(output_image, f"{primary_emotion[0]}: {primary_emotion[1]:.2f}",
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        face_emotions.append({
                            'primary_emotion': primary_emotion[0],
                            'confidence': primary_emotion[1],
                            'all_emotions': emotions_dict
                        })

    ##### FER Analysis #####


        
        if len(face_regions) == 0:
            # If no faces detected, try to use FER's built-in detection
            emotions = self.emotion_detector.detect_emotions(image_rgb)
            
            if len(emotions) > 0:
                for emotion_data in emotions:
                    box = emotion_data['box']
                    x, y, w, h = box
                    emotions_dict = emotion_data['emotions']
                    
                    # Find primary emotion
                    primary_emotion = max(emotions_dict.items(), key=lambda x: x[1])
                    
                    # Draw rectangle on image
                    cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(output_image, f"{primary_emotion[0]}: {primary_emotion[1]:.2f}", 
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    face_emotions.append({
                        'primary_emotion': primary_emotion[0],
                        'confidence': primary_emotion[1],
                        'all_emotions': emotions_dict
                    })
        else:
            # Process each detected face
            for i, (x1, y1, x2, y2) in enumerate(face_regions):
                face_img = image_rgb[y1:y2, x1:x2]
                
                # Skip if face is too small
                if face_img.size == 0 or face_img.shape[0] < 20 or face_img.shape[1] < 20:
                    continue
                
                # Analyze emotions using FER
                emotion_data = self.emotion_detector.detect_emotions(face_img)
                
                if len(emotion_data) > 0:
                    emotions_dict = emotion_data[0]['emotions']
                    
                    # Find primary emotion
                    primary_emotion = max(emotions_dict.items(), key=lambda x: x[1])
                    
                    # Draw rectangle and emotion on image
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(output_image, f"{primary_emotion[0]}: {primary_emotion[1]:.2f}", 
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    face_emotions.append({
                        'primary_emotion': primary_emotion[0],
                        'confidence': primary_emotion[1],
                        'all_emotions': emotions_dict
                    })
        
        # Save the processed image
        output_filename = f"processed_{uuid.uuid4().hex}.jpg"
        output_path = os.path.join('static/uploads', output_filename)
        cv2.imwrite(output_path, output_image)
        
        return {
            'image_path': f"uploads/{output_filename}",
            'face_count': len(face_emotions),
            'face_emotions': face_emotions
        }


class VideoEmotionAnalyzer(BaseVisionModel):
    def __init__(self):
        super().__init__()
        
        # Initialize models
        self.models = {
            'FER-2013': 'fer2013',
            'FER+': 'ferplus',
            'HSE': 'hse',  # Human Sentiment Estimation
            'DeepFace': 'deepface'
        }
        
        # Default to FER
        self.emotion_detector = FER(mtcnn=False)  # We already have MTCNN
        self.current_model = 'FER-2013'
    
    def get_available_models(self):
        """Return list of available models"""
        return list(self.models.keys())
    
    def set_model(self, model_name):
        """Change emotion detection model"""
        if model_name in self.models:
            self.current_model = model_name
            # In a real implementation, we would actually load different models here
    
    def analyze_frame(self, frame):
        """Analyze emotions in a video frame"""
        # Convert BGR to RGB (OpenCV uses BGR, but our models expect RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create copy for visualization
        output_frame = frame.copy()
        
        # Detect faces
        face_regions = self.detect_faces(frame_rgb)
        
        # Initialize results
        emotion_results = {
            'face_count': len(face_regions),
            'faces': []
        }

 
        if self.current_model == 'DeepFace':
            try:
                analysis = DeepFace.analyze(img_path=frame_rgb, actions=['emotion'], enforce_detection=False)

                # Normalize result to list
                if isinstance(analysis, dict):
                    analysis = [analysis]

                for result in analysis:
                    dominant_emotion = result['dominant_emotion']
                    emotion_scores = result['emotion']
                    region = result.get('region', {})

                    # Draw bounding box if region is valid
                    if all(key in region for key in ('x', 'y', 'w', 'h')):
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(output_frame, f"{dominant_emotion}: {emotion_scores[dominant_emotion]:.2f}",
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    emotion_results['faces'].append({
                        'primary_emotion': dominant_emotion,
                        'confidence': float(emotion_scores[dominant_emotion]) / 100.0,
                        'all_emotions': {k: float(v) / 100.0 for k, v in emotion_scores.items()}
                    })

                emotion_results['face_count'] = len(analysis)

            except Exception as e:
                print(f"DeepFace video analysis failed: {e}")



        
        # If no faces detected with MTCNN, try FER's detector
        if len(face_regions) == 0:
            # Use FER's built-in detection
            emotions = self.emotion_detector.detect_emotions(frame_rgb)
            
            if len(emotions) > 0:
                for emotion_data in emotions:
                    box = emotion_data['box']
                    x, y, w, h = box
                    emotions_dict = emotion_data['emotions']
                    
                    # Find primary emotion
                    primary_emotion = max(emotions_dict.items(), key=lambda x: x[1])
                    
                    # Draw rectangle on frame
                    cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(output_frame, f"{primary_emotion[0]}: {primary_emotion[1]:.2f}", 
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    emotion_results['faces'].append({
                        'primary_emotion': primary_emotion[0],
                        'confidence': primary_emotion[1],
                        'all_emotions': emotions_dict
                    })
        else:
            # Process each detected face
            for i, (x1, y1, x2, y2) in enumerate(face_regions):
                face_img = frame_rgb[y1:y2, x1:x2]
                
                # Skip if face is too small
                if face_img.size == 0 or face_img.shape[0] < 20 or face_img.shape[1] < 20:
                    continue
                
                # Analyze emotions using FER
                emotion_data = self.emotion_detector.detect_emotions(face_img)
                
                if len(emotion_data) > 0:
                    emotions_dict = emotion_data[0]['emotions']
                    
                    # Find primary emotion
                    primary_emotion = max(emotions_dict.items(), key=lambda x: x[1])
                    
                    # Draw rectangle and emotion on frame
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(output_frame, f"{primary_emotion[0]}: {primary_emotion[1]:.2f}", 
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    emotion_results['faces'].append({
                        'primary_emotion': primary_emotion[0],
                        'confidence': primary_emotion[1],
                        'all_emotions': emotions_dict
                    })
        
        # Add ROI box for demonstration
        height, width = output_frame.shape[:2]
        roi_x1, roi_y1 = int(width * 0.25), int(height * 0.25)
        roi_x2, roi_y2 = int(width * 0.75), int(height * 0.75)
        cv2.rectangle(output_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
        
        return output_frame, emotion_results