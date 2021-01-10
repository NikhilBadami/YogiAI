import pickle
import numpy as np
import mediapipe as mp
mp_pose = mp.solutions.pose
import cv2
import tensorflow as tf
import os

def load_dataset(
        dataset_label,
        data_path,
        class_labels,
        save_pickle=False,
        read_pickle=False
    ):
    """
    :param dataset_label: Train or Test
    :param data_path: Path to data directory
    :param class_labels: Dictionary of class labels and their 1-hot encoded index
    :param save_pickle: whether or not to save the processed data as a pickled file
    :param read_pickle: whether func should read data from file or process raw image data
    :return: tuple of lists containing data samples and corresponding labels
    """

    # Check to see if dataset should be loaded from file
    if read_pickle:
        try:
            return load_dataset_from_pickle(data_path, dataset_label)
        except:
            print(f"Err: pickled data does not exist. Loading data from images")

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
        save_dataset_to_pickle(data_path, dataset_label, data, labels)

    return np.array(data), np.array(labels)

def load_dataset_from_pickle(data_path, dataset_label):
    filename = f"{data_path}pickled_data/{dataset_label}"
    print(f"Loading {dataset_label} data from pickle")

    data_dict = None
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f)
    data = data_dict['data']
    labels = data_dict['labels']
    return np.array(data), np.array(labels)

def save_dataset_to_pickle(data_path, dataset_label, data, labels):
    data_dict = {
        'data': data,
        'labels': labels
    }
    filename = f"{data_path}pickled_data/{dataset_label}"
    # Clear the contents of the file
    open(filename, 'wb').close()

    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"saved {filename}")

def load_data(config, data_path, class_labels):
    train_data, train_labels = load_dataset(
        "Train",
        data_path,
        class_labels,
        save_pickle=config["save_pickle"],
        read_pickle=config["read_pickle"]
    )
    test_data, test_labels = load_dataset(
        "Test",
        data_path,
        class_labels,
        save_pickle=config["save_pickle"],
        read_pickle=config["read_pickle"]
    )
    val_data, val_labels = load_dataset(
        "Validation",
        data_path,
        class_labels,
        save_pickle=config["save_pickle"],
        read_pickle=config["read_pickle"]
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(32)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(32)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels)).batch(32)

    return train_dataset, test_dataset, val_dataset
