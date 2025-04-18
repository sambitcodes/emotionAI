import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flair.models import TextClassifier
from flair.data import Sentence
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')


login(token=HF_TOKEN, add_to_git_credential=True)


class TextEmotionAnalyzer:
    def __init__(self):
        # Initialize model options
        self.models = {
            'TextBlob': 'textblob',
            'VADER': 'vader',
            'Flair': 'flair',
            'BERT Emotion': 'bert-base-uncased-emotion',
            'RoBERTa Emotion': 'roberta-base-emotion'
        }
        
        # Load the models lazily
        self.loaded_models = {}
        
        # Default to first model
        self._load_model(list(self.models.keys())[0])
    
    def _load_model(self, model_name):
        """Load the specified model"""
        if model_name in self.loaded_models:
            # Model already loaded
            self.current_model_name = model_name
            return
        
        self.current_model_name = model_name
        model_id = self.models.get(model_name, list(self.models.values())[0])
        
        # Load model based on type
        if model_id == 'textblob':
            # TextBlob doesn't need pre-loading
            self.loaded_models[model_name] = 'textblob'
        
        elif model_id == 'vader':
            # Initialize VADER
            try:
                import nltk
                nltk.download('vader_lexicon', quiet=True)
                self.loaded_models[model_name] = SentimentIntensityAnalyzer()
            except Exception as e:
                print(f"Failed to load VADER: {e}")
                self.loaded_models[model_name] = None
        
        elif model_id == 'flair':
            # Initialize Flair sentiment classifier
            try:
                self.loaded_models[model_name] = TextClassifier.load('en-sentiment')
            except Exception as e:
                print(f"Failed to load Flair: {e}")
                self.loaded_models[model_name] = None
        
        elif 'bert' in model_id or 'roberta' in model_id:
            # For BERT and RoBERTa, we'll use the transformers pipeline
            try:
                if model_id == 'bert-base-uncased-emotion':
                    model_path = 'bhadresh-savani/bert-base-uncased-emotion'

                elif model_id == 'roberta-base-emotion':
                    model_path = 'j-hartmann/emotion-english-distilroberta-base'

                else:
                    model_path = model_id
                
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
                model = AutoModelForSequenceClassification.from_pretrained(model_path, token=HF_TOKEN)
                
                self.loaded_models[model_name] = {
                    'tokenizer': tokenizer,
                    'model': model,
                    'path': model_path
                }
                
                # Set emotion labels based on model
                if model_path == 'bhadresh-savani/bert-base-uncased-emotion':
                    self.loaded_models[model_name]['labels'] = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
                elif model_path == 'j-hartmann/emotion-english-distilroberta-base':
                    self.loaded_models[model_name]['labels'] = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
                else:
                    # Default labels if not specified
                    self.loaded_models[model_name]['labels'] = ['negative', 'neutral', 'positive']
                
            except Exception as e:
                print(f"Failed to load {model_id}: {e}")
                self.loaded_models[model_name] = None
    
    def get_available_models(self):
        """Return list of available models"""
        return list(self.models.keys())
    
    def analyze(self, text, model_name=None):
        """Analyze text and return emotion predictions"""
        if model_name and model_name != self.current_model_name:
            self._load_model(model_name)
        
        current_model = self.loaded_models.get(self.current_model_name)
        
        if self.current_model_name == 'TextBlob':
            # TextBlob sentiment analysis
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # Map polarity to sentiment categories
            if polarity > 0.1:
                primary_emotion = 'positive'
                confidence = min(abs(polarity) * 2, 1.0)  # Scale to 0-1
            elif polarity < -0.1:
                primary_emotion = 'negative'
                confidence = min(abs(polarity) * 2, 1.0)  # Scale to 0-1
            else:
                primary_emotion = 'neutral'
                confidence = 1.0 - min(abs(polarity) * 5, 0.8)  # Higher confidence for values closer to 0
            
            # Create emotions dictionary
            emotions = {
                'positive': max(0, polarity) * 2 if polarity > 0 else 0,
                'neutral': 1.0 - min(abs(polarity) * 2, 0.8),
                'negative': max(0, -polarity) * 2 if polarity < 0 else 0
            }
            
            return {
                'primary_emotion': primary_emotion,
                'confidence': confidence,
                'all_emotions': emotions
            }
        
        elif self.current_model_name == 'VADER':
            # VADER sentiment analysis
            if current_model:
                scores = current_model.polarity_scores(text)
                
                # Map scores to emotions
                emotions = {
                    'negative': scores['neg'],
                    'neutral': scores['neu'],
                    'positive': scores['pos']
                }
                
                # Get primary emotion
                primary_emotion = max(emotions.items(), key=lambda x: x[1])
                
                return {
                    'primary_emotion': primary_emotion[0],
                    'confidence': primary_emotion[1],
                    'all_emotions': emotions
                }
            else:
                return self._get_error_response()
        
        elif self.current_model_name == 'Flair':
            # Flair sentiment analysis
            if current_model:
                sentence = Sentence(text)
                current_model.predict(sentence)
                
                # Get sentiment and score
                label = sentence.labels[0]
                sentiment = label.value.lower()  # 'POSITIVE' or 'NEGATIVE'
                score = label.score  # confidence score
                
                # Map to common emotions format
                emotions = {
                    'positive': score if sentiment == 'positive' else 0.0,
                    'neutral': 0.0,
                    'negative': score if sentiment == 'negative' else 0.0
                }
                
                return {
                    'primary_emotion': sentiment,
                    'confidence': score,
                    'all_emotions': emotions
                }
            else:
                return self._get_error_response()
        
        elif self.current_model_name == 'BERT Emotion' or self.current_model_name == 'RoBERTa Emotion':
            # Transformer-based emotion analysis
            if current_model:
                # Tokenize input
                tokenizer = current_model['tokenizer']
                model = current_model['model']
                labels = current_model['labels']
                
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                
                # Get predictions
                with torch.no_grad():
                    outputs = model(**inputs)
                    
                # Get probabilities using softmax
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].numpy()
                
                # Create emotion dictionary
                emotions = {label: float(prob) for label, prob in zip(labels, probs)}
                
                # Get primary emotion
                primary_emotion = max(emotions.items(), key=lambda x: x[1])
                
                return {
                    'primary_emotion': primary_emotion[0],
                    'confidence': float(primary_emotion[1]),
                    'all_emotions': emotions
                }
            else:
                return self._get_error_response()
        
        else:
            return self._get_error_response()
    
    def _get_error_response(self):
        """Return error response when model is not available"""
        return {
            'primary_emotion': 'error',
            'confidence': 0.0,
            'all_emotions': {
                'error': 1.0,
                'positive': 0.0,
                'neutral': 0.0,
                'negative': 0.0
            }
        }