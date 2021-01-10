import mediapipe as mp
mp_pose = mp.solutions.pose
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_model():
    """
    Create the Keras model
    :return: model: Keras model
    """
    model = keras.Sequential()
    model.add(
        layers.Conv1D(
            filters=16,
            kernel_size=3,
            activation=keras.activations.relu,
            padding="same",
            input_shape=(33, 2)
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.MaxPooling1D())
    model.add(
        layers.Conv1D(
            filters=16,
            kernel_size=3,
            activation=keras.activations.relu,
            padding="same",
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(5, activation=keras.activations.softmax))
    # model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    )
    return model

def get_label_from_prediction(prediction, class_labels):
    """
    Gets the label from the 1-hot encoded prediction
    :param prediction: 1-hot encoded output from model
    :return: label: name of predicted pose
    """
    prediction_np = prediction.numpy()
    predicted_label = np.argmax(prediction_np)
    label = 'No pose detected'
    for class_name in class_labels.keys():
        if predicted_label == class_labels[class_name]:
            label = class_name
            break
    return label

def predict_with_static_image(
        model,
        class_labels,
        data_path,
):
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    for class_name in class_labels.keys():
        path = f"{data_path}{class_name}/Test"
        filenames = os.listdir(path)
        idx = random.randint(0, len(filenames) - 1)
        filename = filenames[idx]
        if not filename.endswith(".jpg"):
            while not filename.endswith(".jpg"):
                idx = random.randint(0, len(filenames))
                filename = filenames[idx]
        filepath = f"{path}/{filename}"
        image = cv2.imread(filepath)
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw skeleton on image
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('', annotated_image)
        cv2.waitKey()

        # Extract x,y coordinates of key poitns
        sample = []
        for lm in results.pose_landmarks.landmark:
            sample.append((lm.x, lm.y))

        # get prediction for skeleton
        print(class_name)
        prediction = model(np.array(sample)[np.newaxis, :, :])
        print(prediction)

def predict_with_video(model, class_labels):
    # Test net on video
    cap = cv2.VideoCapture(0)
    if not cap:
        print("Cannot open camera")

    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    label = ''
    while True:
        # Get frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame")
            break

        # Get skeleton in frame
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            continue
        sample = []
        for lm in results.pose_landmarks.landmark:
            sample.append((lm.x, lm.y))

        # Predict pose and get label
        prediction = model(np.array(sample)[np.newaxis, :, :])
        label = get_label_from_prediction(prediction, class_labels)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # Write class name on image
        cv2.putText(
            frame,
            label,
            tuple((50, 100)),
            cv2.FONT_HERSHEY_SIMPLEX,
            4,
            tuple((255, 0, 0)),
            2
        )

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def train_model(model, config, train_dataset, val_dataset):
    history = model.fit(train_dataset, epochs=100, validation_data=val_dataset)
    model.save("/Users/nikhilbadami/Pose Estimation/YogiAI/saved_models/")
    if config["display_stats"]:
        # summarize history for acc
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
