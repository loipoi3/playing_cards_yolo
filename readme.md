# YOLO Playing Card Detection

## Overview
This project focuses on training a YOLO (You Only Look Once) model to detect playing cards in images. The YOLO architecture is a popular object detection algorithm that uses a single neural network to predict bounding boxes and class probabilities directly from full images in real-time.

## Data
To train the YOLO model for playing card detection, you will need a labeled dataset of images containing playing cards. The dataset should include images with bounding box annotations around each playing card, specifying the card's class (e.g., "Ace of Spades," "Queen of Hearts," etc.). It is essential to have a diverse and representative dataset for optimal model performance. I used this dataset: https://www.kaggle.com/datasets/andy8744/playing-cards-object-detection-dataset.

## Model Architecture
The YOLO model architecture used in this project is based on the YOLOv8 (You Only Look Once) variant. YOLOv8 is a widely adopted version of the YOLO algorithm known for its accuracy and efficiency.

The YOLOv8 architecture consists of a backbone network (such as Darknet-53) for feature extraction, followed by several detection layers that predict bounding boxes and class probabilities at different scales. The model leverages anchor boxes and multi-scale predictions to detect objects of various sizes.

## Training
1. Data Preparation: Preprocess your dataset by resizing the images to a consistent input size and converting the annotations to YOLO format. The YOLO format requires bounding box annotations to be represented as normalized coordinates (center_x, center_y, width, height) relative to the image dimensions.

2. Model Configuration: Configure the YOLO model architecture and adjust hyperparameters based on your specific requirements. This includes setting the number of classes, anchor box sizes, training batch size, learning rate, and other related parameters.

3. Training Process: Train the YOLO model on your labeled dataset. During training, the model learns to predict bounding box coordinates and class probabilities. The training process typically involves optimizing a loss function that combines localization loss and classification loss. For training model i used this command:
```bash
yolo detect train data=./data.yaml model=yolov8n.pt epochs=120 imgsz=416 batch=16 flipud=0.0 fliplr=0.0
```

4. Model Evaluation: Evaluate the trained model's performance using appropriate metrics such as mean average precision (mAP) and intersection over union (IoU). These metrics help assess the model's ability to detect playing cards accurately.

5. Inference: Once the model is trained and evaluated, you can use it for inference on new images. The trained YOLO model can detect playing cards in real-time, providing bounding box predictions and associated class probabilities.

## Usage
To use the trained model for Playing Cards Detection, follow the instructions below:

1. First clone the repository. To do this, open a terminal, go to the directory where you want to clone the project and then enter the command:
```bash
git clone https://github.com/loipoi3/playing_cards_yolo.git
```
2. Next go to folder with project and run this command:
```bash
docker-compose up --build
```
3. And the last thing, open this link in your browser http://localhost:8501, that's all, now you can use the detector.

## Results
The performance of the YOLO model can be measured using metrics like mean average precision (mAP), precision, recall, and intersection over union (IoU). These metrics provide insights into the model's accuracy and detection capabilities.

It is crucial to evaluate the model on a separate validation or test set to assess its generalization and ensure it performs well on unseen data.

The model was trained for 20 epochs with a lerning rate of 0.01 and an SGD optimizer with a momentum of 0.9, after the last epoch the mAP50-95 metric was 0.95101, you can see more about the metrics and losses in the file ./runs/detect/train/results.csv.

## Conclusion
In this project, we trained a custom YOLO model for playing card detection. By following the steps outlined above, you can create your own playing card detection system using YOLO. Remember to use a diverse and representative dataset, appropriately configure the model, and evaluate its performance to achieve accurate and reliable results.

Please refer to the project documentation and code for detailed implementation instructions and examples. Happy detecting!

## Author
This YOLO Playing Card Detection project was developed by Dmytro Khar. If you have any questions or need further assistance, please contact qwedsazxc8250@gmail.com.
