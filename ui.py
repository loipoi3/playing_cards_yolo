import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import cv2
import numpy as np


def send_prediction_request(image):
    url = 'http://192.168.0.108:5000/detect'
    image_byte_arr = io.BytesIO()
    image.save(image_byte_arr, format='JPEG')
    image_byte_arr = image_byte_arr.getvalue()
    files = {'file': image_byte_arr}
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        result = response.json()
        return result.get('data')
    except requests.exceptions.RequestException as e:
        return f'Error: {e}'


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
        font_size = 0.25
        font_thickness = 1
        text_offset = 2

        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, thickness)
        cv2.putText(image, label, (int(x_min), int(y_min - text_offset)), cv2.FONT_HERSHEY_SIMPLEX, font_size, color,
                    font_thickness, cv2.LINE_AA, False)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    st.image(image, caption='Object Detection', use_column_width=True)


def main():
    st.title("Object Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        res = send_prediction_request(image)
        if isinstance(res, str) and res.startswith('Error'):
            st.error(res)
        else:
            visualize_detection(image, res)


if __name__ == '__main__':
    st.set_page_config(page_title="Object Detection")
    main()
