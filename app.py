from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import shutil
from werkzeug.utils import secure_filename
import uuid
from image_processor import process_image, setup_directories
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# App configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_DIR'] = 'output'
app.config['PREPROCESSED_DIR'] = os.path.join(app.config['OUTPUT_DIR'], 'preprocessed')
app.config['GEMINI_OUTPUT_DIR'] = os.path.join(app.config['OUTPUT_DIR'], 'gemini_edited')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

# Ensure directories exist
setup_directories(app.config['UPLOAD_FOLDER'], 
                 app.config['OUTPUT_DIR'],
                 app.config['PREPROCESSED_DIR'],
                 app.config['GEMINI_OUTPUT_DIR'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_images():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files[]')
    prompt = request.form.get('prompt', '')
    api_key = request.form.get('api_key', os.environ.get('GOOGLE_API_KEY', ''))
    
    if not api_key:
        return jsonify({'error': 'API key is required'}), 400
    
    if not files or len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400
    
    if len(files) > 10:
        return jsonify({'error': 'Maximum 10 files allowed'}), 400
    
    # Process each image
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            # Create unique filename
            original_filename = secure_filename(file.filename)
            filename = f"{uuid.uuid4()}_{original_filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process the image
            try:
                result = process_image(file_path, prompt, api_key, 
                                      app.config['PREPROCESSED_DIR'],
                                      app.config['GEMINI_OUTPUT_DIR'])
                result['original_filename'] = original_filename
                results.append(result)
            except Exception as e:
                results.append({
                    'original_filename': original_filename,
                    'error': str(e),
                    'success': False
                })
    
    return jsonify({'results': results})

@app.route('/output/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_DIR'], filename, as_attachment=True)

@app.route('/view/<path:filename>')
def view_file(filename):
    return send_from_directory(app.config['OUTPUT_DIR'], filename)

if __name__ == '__main__':
    app.run(debug=True)