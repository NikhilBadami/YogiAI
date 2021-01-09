import mediapipe as mp
mp_pose = mp.solutions.pose
import os
import cv2
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import pickle

class_labels = {
    "Warrior_I": 0,
    "Warrior_II": 1,
    "Tree": 2,
    "Triangle": 3,
    "Standing Splits": 4
}

data_path = "/Users/nikhilbadami/Pose Estimation/YogiAI/data copy/"


def load_dataset(dataset_label, save_pickle=False, read_pickle=False):
    """
    :param dataset_label: Train or Test
    :param save_pickle: whether or not to save the processed data as a pickled file
    :param read_pickle: whether func should read data from file or process raw image data
    :return: tuple of lists containing data samples and corresponding labels
    """

    # Check to see if dataset should be loaded from file
    if read_pickle:
        filename = f"{data_path}pickled_data/{dataset_label}"
        try:
            print(f"Loading {dataset_label} data from pickle")
            data_dict = None
            with open(filename, 'rb') as f:
                data_dict = pickle.load(f)
            data = data_dict['data']
            labels = data_dict['labels']
            return np.array(data), np.array(labels)
        except:
            print(f"Err: {filename} does not exist. Loading data from images")

    # Load raw imags and extract pose skeleton. Will result in 32 keypoints
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    data = []
    labels = []
    for class_name in class_labels.keys():
        path = f"{data_path}{class_name}/{dataset_label}"
        print(f"Loading {dataset_label} data for {class_name}")
        for filename in os.listdir(path):
            if filename.endswith(".jpg"):
                image = cv2.imread(f"{path}/{filename}")
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if not results.pose_landmarks:
                    # Ignore any images that don't produce skeletons
                    continue

                # Combine landmarks into datasample
                sample = []
                for lm in results.pose_landmarks.landmark:
                    # Create sample which is M x 2 where M is the number of keypoints detected and their
                    # x and y coordinates
                    sample.append((lm.x, lm.y))

                data.append(sample)
                # Create label sample
                label_sample = np.zeros(5)
                label_sample[class_labels[class_name]] = 1
                labels.append(label_sample)

    pose.close()

    # Check to see if this data should be pickled
    if save_pickle:
        data_dict = {
            'data': data,
            'labels': labels
        }
        filename = f"{data_path}pickled_data/{dataset_label}"
        with open(filename, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"saved {filename}")

    return np.array(data), np.array(labels)

def create_model():
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
    model.add(layers.Flatten())
    model.add(layers.Dense(5, activation=keras.activations.softmax))
    # model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    )
    return model


if __name__ == "__main__":
    # Load training and test datasets
    train_data, train_labels = load_dataset("Train", save_pickle=True, read_pickle=True)
    test_data, test_labels = load_dataset("Test", save_pickle=True, read_pickle=True)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(32)

    # Load CNN
    model = create_model()

    # Train the model
    model.fit(train_dataset, epochs=100)

    # Evaluate model
    loss, acc = model.evaluate(test_dataset)
    print(f"Loss: {loss}")
    print(f"Acc: {acc}")
