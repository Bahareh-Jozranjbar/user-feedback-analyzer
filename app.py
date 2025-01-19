from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from transformers import pipeline
import pandas as pd
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

# Load your fine-tuned sentiment analysis pipeline
sentiment_model = pipeline(model="BaharehJozranjbar/finetuning-sentiment-model-3000-samples")

# Ensure the uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Perform sentiment analysis
    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        return f"Error reading file: {str(e)}", 400

    if 'text' not in data.columns:
        return "The uploaded file must have a 'text' column.", 400

    # Analyze sentiments
    data['sentiment'] = data['text'].apply(lambda x: sentiment_model(x)[0]['label'])
    data['score'] = data['text'].apply(lambda x: sentiment_model(x)[0]['score'])

    # Save results to a new file
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{filename}")
    data.to_csv(result_path, index=False)

    return redirect(url_for('result', filename=f"result_{filename}"))

@app.route('/result/<filename>')
def result(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    data = pd.read_csv(filepath)

    # Display the first few rows of the results
    return render_template('result.html', tables=[data.head().to_html(classes='data')], titles=data.columns.values)

if __name__ == "__main__":
    app.run(debug=True)
