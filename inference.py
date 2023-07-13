from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

model = YOLO('./runs/detect/train2/weights/best.pt')


@app.route('/detect', methods=['POST'])
def run_inference():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'})

    file = request.files['file']
    image = Image.open(file.stream)
    results = model(image)
    serialized_results = {
        'data': results[0].boxes.data.tolist()
    }

    return jsonify(serialized_results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
