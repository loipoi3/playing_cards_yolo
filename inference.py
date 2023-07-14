import torch.cuda
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import cv2
import os

app = Flask(__name__)

model = YOLO('./runs/detect/train2/weights/best.pt')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@app.route('/detect_images', methods=['POST'])
def run_inference_img():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'})

    file = request.files['file']
    image = Image.open(file.stream)

    results = model(image)
    serialized_results = {
        'data': results[0].boxes.data.tolist()
    }

    return jsonify(serialized_results)


@app.route('/detect_videos', methods=['POST'])
def run_inference_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'})

    video_dir = './temp/'
    os.makedirs(video_dir, exist_ok=True)

    file = request.files['file']
    video_path = './temp/video.mp4'
    output_path = './temp/output.mp4'

    # Save the video file
    file.save(video_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a video writer for the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

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

        # Draw bounding boxes on the frame
        for bbox in bboxes:
            x_min, y_min, x_max, y_max, prob_class, class_id = bbox
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
            label = f"{class_id}: {prob_class:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            cv2.rectangle(frame, (int(x_min), int(y_min) - text_size[1]),
                          (int(x_min) + text_size[0], int(y_min)), color, cv2.FILLED)
            cv2.putText(frame, label, (int(x_min), int(y_min)), font, font_scale, (0, 0, 0), font_thickness)

        # Write the frame with bounding boxes to the output video
        out.write(frame)

    # Release the video capture and writer
    cap.release()
    out.release()

    # Return the output video file
    with open(output_path, 'rb') as f:
        video_data = f.read()

    return video_data, 200, {'Content-Type': 'video/mp4'}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
