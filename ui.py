import streamlit as st
import requests
from PIL import Image
import io
import shutil


def send_prediction_request_img(image):
    url = 'http://192.168.0.108:5000/detect_images'
    image_byte_arr = io.BytesIO()
    image.save(image_byte_arr, format='JPEG')
    image_byte_arr = image_byte_arr.getvalue()
    files = {'file': image_byte_arr}
    try:
        response = requests.post(url, files=files)  # Pass the headers argument
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        return f'Error: {e}'


def main():
    st.title("Object Detection")
    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        if uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file)
            res = send_prediction_request_img(image)
            if isinstance(res, str) and res.startswith('Error'):
                st.error(res)
            else:
                st.image(res, caption='Object Detection', use_column_width=True)
        elif uploaded_file.type.startswith('video'):
            video = uploaded_file.getvalue()
            v = st.video(video)
            if st.button('Run'):
                v.empty()
                t = st.empty()
                t.markdown('Running...')
                predicted = requests.post(f"http://192.168.0.108:5000/detect_videos", files={'file': uploaded_file})
                if predicted.status_code == 200:
                    output_video = predicted.content

                    # Display the output video in Streamlit
                    st.video(output_video)
                    shutil.rmtree('./temp')
                else:
                    st.error(f"Error: {predicted.status_code} - {predicted.content}")

if __name__ == '__main__':
    st.set_page_config(page_title="Object Detection")
    main()
