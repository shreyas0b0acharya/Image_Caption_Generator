from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import os
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_LEN'] = 37 
app.config['DEBUG'] = True 


# Load InceptionV3 for feature extraction

print("Loading InceptionV3 for feature extraction...")
inception_model = InceptionV3(weights='imagenet')
inception_model = Model(inputs=inception_model.inputs, 
                        outputs=inception_model.layers[-2].output)


# Load Captioning Model and Tokenizer

print("Loading captioning model and tokenizer...")
model = load_model('model/image_captioning_model.keras')

with open('model/tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_json = f.read()  
    tokenizer = tokenizer_from_json(tokenizer_json)

# Get vocabulary size

word_to_idx = tokenizer.word_index
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
vocab_size = len(word_to_idx) + 1  # +1 for padding


# Image Preprocessing & Feature Extraction

def extract_image_features(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)  # InceptionV3 specific preprocessing
    features = inception_model.predict(img, verbose=0)
    return features.flatten()


# Generate Caption Function

def generate_caption(image_path, max_len):
    image_features = extract_image_features(image_path)
    caption = 'start'
    for _ in range(max_len):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')[0]
        
        pred = model.predict(
            [np.array([image_features]), np.array([sequence])], 
            verbose=0
        )
        
        pred_id = np.argmax(pred[0])
        word = idx_to_word.get(pred_id, '')

        if word == 'end' or word == '' or len(caption.split()) > max_len:
            break
            
        caption += ' ' + word
    
    # Clean up the caption
    result = caption.replace('start', '').replace('end', '').strip()
    return result if result else "Unable to generate caption"


# Flask Routes

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = None
    image_url = None
    error = None

    if request.method == 'POST':
        if 'image' not in request.files:
            error = 'No file part in the request'
        else:
            file = request.files['image']
            if file.filename == '':
                error = 'No file selected'
            elif file:
                # Save uploaded image
                filename = file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image_url = url_for('static', filename=f'uploads/{filename}')
                
                # Generate caption with error handling
                try:
                    caption = generate_caption(filepath, app.config['MAX_LEN'])
                except Exception as e:
                    error = f"Error generating caption: {str(e)}"
                    print(f"Caption generation error: {e}")

    return render_template('index.html', 
                          caption=caption, 
                          image_url=image_url,
                          error=error)


# Run the App

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    print("Starting Flask application...")
    print(f" * Model loaded with vocab size: {vocab_size}")
    print(f" * Max caption length: {app.config['MAX_LEN']}")
    
    app.run(debug=app.config['DEBUG'])