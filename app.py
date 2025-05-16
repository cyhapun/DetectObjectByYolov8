from flask import Flask, request, render_template
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

model = YOLO('yolov8s.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    files = request.files.getlist('file')
    if not files or files[0].filename == '':
        # Hiển thị thông báo trên trang chủ nếu không có file nào
        return render_template('index.html', message="Vui lòng chọn ít nhất một tệp để tải lên.")
    result_filenames = []
    for file in files:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        results = model(file_path)
        result_img = results[0].plot()
        output_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        cv2.imwrite(output_path, result_img)
        result_filenames.append(filename)
    if not result_filenames:
        return render_template('index.html', message="Không có tệp hợp lệ được tải lên.")
    return render_template('result.html', filenames=result_filenames)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)