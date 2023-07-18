import torch.cuda
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import cv2
import os
import numpy as np
import io

app = Flask(__name__)

model = YOLO('./runs/detect/train/weights/best.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

def visualize_detection(image, detections):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    class_names = ['10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s', '4c', '4d', '4h', '4s',
                   '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s', '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s', '9c',
                   '9d', '9h', '9s', 'Ac', 'Ad', 'Ah', 'As', 'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh', 'Ks', 'Qc', 'Qd',
                   'Qh', 'Qs']

    for bbox in detections:
        x_min, y_min, x_max, y_max, prob_class, class_id = bbox
        class_id = int(class_id)
        label = f"{class_names[class_id]}: {prob_class:.2f}"
        color = (0, 0, 255)
        thickness = 1
        font_size = 0.3
        font_thickness = 1
        text_offset = 2

        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_thickness)
        cv2.rectangle(image, (int(x_min), int(y_min - text_size[1] - text_offset)),
                      (int(x_min) + text_size[0], int(y_min)), (0, 0, 255), cv2.FILLED)

        # Draw white text above the bounding box
        cv2.putText(image, label, (int(x_min), int(y_min - text_offset)), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    (255, 255, 255), font_thickness)

        # Draw the bounding box
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image


@app.route('/detect_images', methods=['POST'])
def run_inference_img():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'})

    file = request.files['file']
    image = Image.open(file.stream)

    results = model(image)

    img = visualize_detection(image, results[0].boxes.data.tolist())
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    return img_byte_arr, 200, {'Content-Type': 'image/jpeg'}


@app.route('/detect_videos', methods=['POST'])
def run_inference_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'})

    file = request.files['file']

    # Save the video file
    file.save(file.filename)

    # Open the video file
    cap = cv2.VideoCapture(file.filename)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a video writer for the output video
    fourcc = cv2.VideoWriter_fourcc('V','P','8','0')
    out = cv2.VideoWriter('output.webm', fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Run inference on the frame using the model
        results = model(image)

        # Get the bounding boxes from the results
        bboxes = results[0].boxes.data.tolist()

        class_names = ['10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s', '4c', '4d', '4h',
                       '4s', '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s', '7c', '7d', '7h', '7s', '8c', '8d', '8h',
                       '8s', '9c', '9d', '9h', '9s', 'Ac', 'Ad', 'Ah', 'As', 'Jc', 'Jd', 'Jh', 'Js', 'Kc', 'Kd', 'Kh',
                       'Ks', 'Qc', 'Qd', 'Qh', 'Qs']

        # Draw bounding boxes on the frame
        for bbox in bboxes:
            x_min, y_min, x_max, y_max, prob_class, class_id = bbox
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
            label = f"{class_names[int(class_id)]}: {prob_class:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.3
            font_thickness = 2
            text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            cv2.rectangle(frame, (int(x_min), int(y_min) - text_size[1]),
                          (int(x_min) + text_size[0], int(y_min)), color, cv2.FILLED)
            cv2.putText(frame, label, (int(x_min), int(y_min)), font, font_scale, (255, 255, 255), font_thickness)

        # Write the frame with bounding boxes to the output video
        out.write(frame)

    # Release the video capture and writer
    cap.release()
    out.release()

    # Return the output video file
    with open('output.webm', 'rb') as f:
        video_data = f.read()

    return video_data, 200, {'Content-Type': 'video/mp4'}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
