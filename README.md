# emoAI - Emotion Analysis Web Application

## Overview

This project is a comprehensive Flask-based web application for emotion analysis across multiple modalities. The application provides a user-friendly interface to analyze emotions in text, images, videos, and Twitter data. The system uses state-of-the-art machine learning and deep learning models to detect and classify emotions, presenting the results with intuitive visualizations.

## Features

The application consists of four main modules:

1. **Text Emotion Analysis**: Analyze emotions in user-provided text input using multiple NLP models (Flair, VADER, TextBlob, BERT, RoBERTa)
2. **Image Emotion Analysis**: Detect faces and analyze emotions in uploaded images using FER and DeepFace
3. **Video Emotion Analysis**: Real-time emotion detection through webcam feed using FER and DeepFace
4. **Twitter Analysis**: Analyze emotions in recent tweets for specified keywords

## Project Structure

```
emotion-analysis-app/
├── app.py                      # Main Flask application
├── requirements.txt            # Project dependencies
├── models/
│   ├── text_models.py          # Text emotion analysis models
│   └── vision_models.py        # Image/video emotion analysis models
├── utils/
│   └── twitter_utils.py        # Twitter API utilities
├── static/
│   ├── css/
│   │   └── style.css           # Custom CSS styles
│   ├── js/
│   │   └── charts.js           # Chart visualization scripts
│   └── uploads/                # Storage for uploaded images
└── templates/
    ├── base.html               # Base template with common structure
    ├── text_analysis.html      # Text analysis interface
    ├── image_analysis.html     # Image analysis interface
    ├── video_analysis.html     # Video analysis interface
    └── twitter_analysis.html   # Twitter analysis interface
```

## Technology Stack

### Backend
- **Flask**: Python web framework to handle HTTP requests and serve pages
- **Text Analysis Libraries**:
  - **Flair**: NLP framework for state-of-the-art text classification
  - **VADER**: Valence Aware Dictionary for sEntiment Reasoning
  - **TextBlob**: Simple NLP library for sentiment analysis
  - **Transformers**: Hugging Face's library for BERT and RoBERTa models
- **Vision Analysis Libraries**:
  - **OpenCV**: Computer vision library for image/video processing
  - **FER**: Facial Emotion Recognition library
  - **DeepFace**: Facial analysis library for emotion detection
- **Tweepy**: Twitter API client for Python

### Frontend
- **HTML/CSS/JavaScript**: Frontend structure, styling, and interactivity
- **Bootstrap 5**: Responsive UI components and layout
- **Chart.js**: Interactive charts for emotion visualization
- **AJAX**: Asynchronous communication with the backend

## How It Works

### Text Emotion Analysis

The text analysis module uses multiple NLP techniques and models to analyze emotions in text. Users can select from five different models, each with its own approach:

#### Backend Process:
1. Text input is preprocessed (cleaning, tokenization)
2. The text is analyzed using the selected model:
   - **Flair**: Context-aware embeddings for emotion classification
   - **VADER**: Rule-based sentiment analysis specialized for social media
   - **TextBlob**: Simple lexicon-based sentiment analysis
   - **BERT**: Deep bidirectional transformers for emotional context understanding
   - **RoBERTa**: Optimized BERT variant with improved training methodology
3. The model produces confidence scores for emotion categories
4. Results are returned to the frontend

#### Detailed Model Explanations:

##### Flair
Flair is an NLP framework that provides state-of-the-art word embeddings and text classification capabilities. For emotion analysis:
- Uses contextual string embeddings that capture semantic meanings
- Implements document embeddings that combine word representations
- Provides pre-trained models for sentiment classification
- Can be fine-tuned on emotion datasets to detect specific emotional states
- Offers both positive/negative sentiment and more granular emotion categories

##### VADER (Valence Aware Dictionary for sEntiment Reasoning)
VADER is a lexicon and rule-based sentiment analysis tool specifically attuned to sentiments expressed in social media:
- Relies on a sentiment lexicon mapping words to sentiment intensity scores
- Incorporates grammatical and syntactical rules to modify sentiment intensity
- Accounts for punctuation, capitalization, degree modifiers, and negations
- Returns compound scores ranging from -1 (extremely negative) to +1 (extremely positive)
- Well-suited for short, informal text snippets

##### TextBlob
TextBlob is a simplified text processing library that provides basic sentiment analysis:
- Uses a pre-trained model based on a corpus of movie reviews
- Returns polarity scores ranging from -1.0 (negative) to 1.0 (positive)
- Also provides subjectivity scores from 0.0 (objective) to 1.0 (subjective)
- Performs simple part-of-speech tagging and noun phrase extraction
- Easy to use but less nuanced than more sophisticated models

##### BERT (Bidirectional Encoder Representations from Transformers)
BERT is a transformer-based language model that revolutionized NLP:
- Pre-trained on massive text corpora to develop deep bidirectional representations
- Fine-tuned on emotion datasets to classify text into emotion categories
- Considers the full context of a word by analyzing surrounding words bidirectionally
- Uses attention mechanisms to focus on relevant parts of the text
- 12-layer architecture with 110M parameters for BERT-base
- Capable of capturing complex emotional nuances in text

##### RoBERTa (Robustly Optimized BERT Pretraining Approach)
RoBERTa is an optimized version of BERT with improved training methodology:
- Trained with larger batches over more data
- Uses dynamic masking patterns for better representation learning
- Removes next sentence prediction objective for more focused training
- Fine-tuned for emotion detection with improved accuracy
- Better handling of subtle emotional expressions and mixed emotions
- 12-layer architecture with 125M parameters

### Image Emotion Analysis

The image analysis module detects faces in uploaded images and analyzes the emotions for each face using FER and DeepFace libraries.

#### Backend Process:
1. The uploaded image is processed using OpenCV
2. Face detection is performed to locate faces in the image
3. Each face is preprocessed (resized, normalized)
4. The emotion classifiers (FER and DeepFace) predict emotions for each face
5. Bounding boxes and emotion labels are drawn on the image
6. The annotated image and emotion scores are returned to the frontend

#### Detailed Model Explanations:

##### FER (Facial Emotion Recognition)
FER is a Python library specifically designed for real-time facial emotion recognition:
- Uses a CNN architecture trained on the FER2013 dataset
- Detects seven emotion categories: angry, disgust, fear, happy, sad, surprise, neutral
- Preprocessing pipeline includes face detection, alignment, and normalization
- Returns confidence scores for each emotion category
- Optimized for real-time applications with reasonable inference speed
- Works well for frontal faces with clear expressions

The CNN architecture in FER typically includes:
- Input layer accepting 48x48 grayscale images
- Multiple convolutional layers with ReLU activations
- Max pooling layers for downsampling
- Dropout layers to prevent overfitting
- Fully connected layers
- Softmax output layer for emotion probabilities

##### DeepFace
DeepFace is a facial analysis framework that provides comprehensive facial attribute analysis:
- Offers multiple pre-trained models for emotion recognition
- Supports various face detection backends (OpenCV, MTCNN, RetinaFace, SSD, Dlib)
- Provides more robust face detection across different angles and lighting conditions
- Analyzes multiple facial attributes beyond emotions (age, gender, race)
- Higher accuracy but potentially slower than lightweight alternatives
- Returns confidence scores across emotion categories

DeepFace's emotion recognition models are typically based on:
- Deep CNN architectures (VGG-Face, Facenet, etc.)
- Transfer learning from face recognition tasks
- Fine-tuning on emotion datasets
- Ensemble approaches combining multiple models for improved accuracy

### Video Emotion Analysis

The video analysis module provides real-time emotion detection through the user's webcam.

#### Backend Process:
1. Webcam feed is captured frame-by-frame using OpenCV
2. For each frame:
   - Faces are detected
   - Region of Interest (ROI) is extracted
   - FER and DeepFace analyze facial emotions
   - Results are overlaid on the video feed
3. Processed frames are streamed back to the browser
4. Emotion scores are continuously updated and visualized

The video analysis uses the same underlying FER and DeepFace models as the image analysis but applies them to video frames in real-time. Optimizations include:
- Frame-rate reduction to balance performance and accuracy
- Region tracking to reduce redundant face detection
- Confidence thresholding to filter unreliable predictions
- Temporal smoothing to reduce prediction jitter

### Twitter Analysis

The Twitter analysis module fetches recent tweets for a given keyword and analyzes the emotions in each tweet.

#### Backend Process:
1. The Twitter API is queried for recent tweets containing the specified keyword
2. For each tweet:
   - Text is extracted and preprocessed
   - Text is analyzed using the selected text emotion analyzer (Flair, VADER, TextBlob, BERT, or RoBERTa)
   - Results are collected and aggregated
3. Individual tweet emotions and overall emotion distribution are returned

## Model Architectures

### Text Emotion Models

1. **Flair Model**:
   - Based on contextual string embeddings
   - Uses stacked embeddings combining different embedding types
   - Document-level classification using hierarchical structure
   - Pre-trained on various text corpora and fine-tuned for emotions
   - Balances accuracy and efficiency for sentiment analysis tasks

2. **VADER Model**:
   - Lexicon and rule-based sentiment analyzer
   - No neural network component - uses dictionary lookups and rules
   - Specifically designed for social media content
   - Extremely fast inference times compared to deep learning models
   - Returns compound scores as well as positive, negative, and neutral proportions

3. **TextBlob Model**:
   - Simple pattern-based analyzer using a trained classifier
   - Based on the Pattern library's sentiment detection
   - Lightweight implementation with minimal dependencies
   - Fast but less nuanced than deep learning approaches
   - Good for basic sentiment polarity but limited for fine-grained emotions

4. **BERT-based Model**:
   - Bidirectional Encoder Representations from Transformers
   - Pre-trained on massive text corpora for understanding context
   - Fine-tuned on emotion datasets with 7 emotion categories
   - 12-layer architecture with 110M parameters
   - Powerful but computationally intensive

5. **RoBERTa-based Model**:
   - Optimized version of BERT with improved training methodology
   - More robust performance across various text styles
   - Fine-tuned specifically for emotion detection
   - 12-layer architecture with 125M parameters
   - State-of-the-art performance for nuanced emotion detection

### Vision Emotion Models

1. **FER (Facial Emotion Recognition)**:
   - CNN-based architecture trained on FER2013 dataset
   - Structure:
     - Input layer (48x48 grayscale images)
     - Multiple convolutional blocks with ReLU activation
     - Max pooling layers for downsampling
     - Dropout layers for regularization
     - Fully connected layers
     - Softmax output for 7 emotion categories
   - Lightweight and optimized for real-time applications
   - Good balance between speed and accuracy

2. **DeepFace**:
   - Comprehensive facial analysis framework
   - Uses deep CNN architectures (VGG-Face, Facenet)
   - Multiple face detection options:
     - MTCNN (Multi-task Cascaded Convolutional Network)
     - RetinaFace (State-of-the-art face detector)
     - OpenCV's DNN or Haar Cascade classifiers
   - Emotion recognition based on transfer learning from face recognition tasks
   - Higher accuracy but more computationally intensive than FER
   - Handles more challenging face angles and lighting conditions

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotion-analysis-app.git
cd emotion-analysis-app
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up Twitter API credentials (for Twitter analysis):
   - Create a file named `.env` in the project root
   - Add your Twitter API credentials:
   ```
   TWITTER_API_KEY=your_api_key
   TWITTER_API_SECRET=your_api_secret
   TWITTER_ACCESS_TOKEN=your_access_token
   TWITTER_ACCESS_SECRET=your_access_secret
   ```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://127.0.0.1:5000`

## Usage Guide

### Text Analysis
1. Navigate to the "Text Analysis" tab
2. Enter the text you want to analyze
3. Select a model from the dropdown (Flair, VADER, TextBlob, BERT, or RoBERTa)
4. Click "Analyze" to process the text
5. View the detected emotion and confidence chart

### Image Analysis
1. Navigate to the "Image Analysis" tab
2. Click "Choose File" to upload an image
3. Select the analysis model (FER or DeepFace)
4. Click "Analyze" to process the image
5. View the detected faces with emotion labels
6. Check the emotion confidence charts for each face

### Video Analysis
1. Navigate to the "Video Analysis" tab
2. Allow camera access when prompted
3. Select the analysis model (FER or DeepFace)
4. Click "Start" to begin real-time analysis
5. Position your face within the frame
6. View real-time emotion detection
7. Click "Stop" to end the session

### Twitter Analysis
1. Navigate to the "Twitter Analysis" tab
2. Enter a keyword or hashtag in the search field
3. Select the text analysis model to use
4. Click "Analyze" to fetch and process tweets
5. View individual tweet emotions in the table
6. Check the overall emotion distribution chart

## Performance Considerations

- **Text Analysis**: 
  - VADER and TextBlob are fastest but less accurate for complex emotions
  - Flair offers a good balance of speed and accuracy
  - BERT and RoBERTa are most accurate but significantly slower
  
- **Image Analysis**:
  - FER is faster but may be less accurate for non-frontal faces
  - DeepFace is more accurate but requires more computational resources
  
- **Video Analysis**: 
  - Requires significant CPU/GPU resources for real-time processing
  - FER recommended for lower-end systems
  - Frame rate can be adjusted to balance performance and accuracy
  
- **Twitter Analysis**: 
  - Limited by Twitter API rate limits (consult API documentation)
  - Processing time increases with the number of tweets analyzed

## Technical Details

### Emotion Categories
The system detects seven standard emotion categories:
- Happy
- Sad
- Angry
- Surprised
- Fearful
- Disgusted
- Neutral

### Confidence Scores
- All models output confidence scores (0-100%) for each emotion category
- Higher scores indicate greater certainty in the emotion classification
- The emotion with the highest confidence score is considered the primary detected emotion

### Model Training

#### Text Models:
- **Flair**: Trained on various text corpora with hierarchical embeddings
- **VADER**: Created using a combination of qualitative and quantitative methods
- **TextBlob**: Trained on movie review data with simple classification
- **BERT**: Pre-trained on BooksCorpus (800M words) and English Wikipedia (2.5B words)
- **RoBERTa**: Pre-trained on 160GB of text including web content and books

#### Vision Models:
- **FER**: Trained on FER2013 dataset (35,887 grayscale images of facial expressions)
- **DeepFace**: Leverages models pre-trained on large face datasets (VGGFace, VGGFace2)

### Comparative Analysis of Text Models

| Model | Strengths | Weaknesses | Best Use Cases |
|-------|-----------|------------|----------------|
| Flair | Good balance of speed/accuracy, context-aware | Moderate resource requirements | General purpose text analysis |
| VADER | Extremely fast, rule-based, no training needed | Limited to sentiment, not fine-grained emotions | Social media, short informal text |
| TextBlob | Simple API, fast, minimal dependencies | Basic sentiment only, less accurate | Quick polarity checks, prototyping |
| BERT | Highly accurate, understands context deeply | Slow, resource intensive | Complex emotional text, nuanced analysis |
| RoBERTa | State-of-the-art accuracy, handles subtlety | Very resource intensive, slowest | Research, highest accuracy needs |

### Comparative Analysis of Vision Models

| Model | Strengths | Weaknesses | Best Use Cases |
|-------|-----------|------------|----------------|
| FER | Fast, lightweight, good for real-time | Less accurate for non-frontal faces | Real-time applications, limited computing resources |
| DeepFace | More accurate, handles difficult angles | Slower, higher resource requirements | High-quality analysis, challenging images |

## Extending the Project

The modular architecture allows for easy extensions:

1. **Additional Models**:
   - Add new model classes to the respective model files
   - Update the frontend to include model selection options

2. **New Features**:
   - Audio emotion analysis through speech processing
   - Batch processing for multiple images or text files
   - Temporal emotion tracking in videos

3. **Performance Improvements**:
   - Model quantization for faster inference
   - GPU acceleration for vision models
   - Caching mechanisms for repeated analyses

## Troubleshooting

Common issues and solutions:

1. **Model loading errors**:
   - Ensure all dependencies are correctly installed
   - Check available disk space for model downloads

2. **Video feed not working**:
   - Verify camera permissions in browser settings
   - Try a different browser if issues persist

3. **Twitter API errors**:
   - Confirm API credentials are correct
   - Check API rate limit status

4. **High memory usage**:
   - For BERT/RoBERTa: Consider using smaller models or Flair/VADER for less powerful systems
   - For DeepFace: Switch to FER for video analysis on systems with limited resources

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for the transformers library (BERT, RoBERTa)
- Flair NLP for the Flair framework
- NLTK team for VADER sentiment analysis
- TextBlob developers for the simplified NLP tools
- FER developers for the facial emotion recognition library
- DeepFace team for the comprehensive facial analysis framework
- OpenCV community for computer vision tools
- FER2013 dataset creators for facial emotion data
