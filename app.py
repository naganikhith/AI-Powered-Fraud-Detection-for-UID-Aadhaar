import os
from flask import Flask, render_template, request
from detection_model.ocr_model import extract_text_fields  # use your exact function

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5 MB limit

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    results = []
    if request.method == 'POST':
        files = request.files.getlist('file')  # support multiple files
        for file in files:
            if file and allowed_file(file.filename):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file.save(filepath)
                # Use your exact OCR extraction function
                result = extract_text_fields(filepath)
                # Simple validity check
                result['is_valid'] = bool(result.get("Aadhaar Number"))
                results.append(result)
            else:
                results.append({"error": "Invalid file"})
    return render_template('index.html', results=results)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
