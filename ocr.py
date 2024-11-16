DEBUG = False

from PIL import Image
import numpy as np

def read_image(path):
    # Read image, convert to grayscale ('L'), and resize to 28x28 pixels
    img = Image.open(path).convert('L')
    img = img.resize((28, 28))  # Resize image to 28x28 to match MNIST data format
    return np.asarray(img)

def write_image(image, path):
    img = Image.fromarray(np.array(image), 'L')
    img.save(path)

DATA_DIR = 'data/'
TEST_DIR = 'test/'
TEST_DATA_FILENAME = DATA_DIR + 'dev-test-images-idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + 'dev-test-labels-idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + 'dev-train-images-idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + 'dev-train-labels-idx1-ubyte'

def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')

def read_images(filename, n_max_images=None):
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4))
        n_columns = bytes_to_int(f.read(4))
        for image_idx in range(n_images):
            image = []
            for row_idx in range(n_rows):
                row = []
                for col_idx in range(n_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images

def read_labels(filename, n_max_labels=None):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for label_idx in range(n_labels):
            label = bytes_to_int(f.read(1))
            labels.append(label)
    return labels

def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]

def extract_features(X):
    return [flatten_list(sample) for sample in X]

def dist(x, y):
    """
    Returns the Euclidean distance between vectors `x` and `y`.
    """
    return sum(
        [
            (bytes_to_int(x_i) - bytes_to_int(y_i)) ** 2
            for x_i, y_i in zip(x, y)
        ]
    ) ** (0.5)

def get_training_distances_for_test_sample(X_train, test_sample):
    return [dist(train_sample, test_sample) for train_sample in X_train]

def get_most_frequent_element(l):
    return max(l, key=l.count)

def knn(X_train, y_train, X_test, k=7):
    y_pred = []
    for test_sample_idx, test_sample in enumerate(X_test):
        print(test_sample_idx, end=' ', flush=True)
        training_distances = get_training_distances_for_test_sample(
            X_train, test_sample
        )
        sorted_distance_indices = [
            pair[0]
            for pair in sorted(
                enumerate(training_distances),
                key=lambda x: x[1]
            )
        ]
        candidates = [
            y_train[idx]
            for idx in sorted_distance_indices[:k]
        ]
        top_candidate = get_most_frequent_element(candidates)
        y_pred.append(top_candidate)
    print()
    return y_pred

import random
from collections import defaultdict

def main():
    n_train = 500
    n_test = 10
    k = 7
    print(f'n_train: {n_train}')
    print(f'n_test: {n_test}')
    print(f'k: {k}')
    
    # Read images and labels
    X_train = read_images(TRAIN_DATA_FILENAME, n_train)
    y_train = read_labels(TRAIN_LABELS_FILENAME, n_train)
    X_test = read_images(TEST_DATA_FILENAME, n_test)
    y_test = read_labels(TEST_LABELS_FILENAME, n_test)

    # Shuffle images and labels together to ensure random order
    combined_train = list(zip(X_train, y_train))
    random.shuffle(combined_train)
    X_train, y_train = zip(*combined_train)  # Unzip after shuffling

    combined_test = list(zip(X_test, y_test))
    random.shuffle(combined_test)
    X_test, y_test = zip(*combined_test)  # Unzip after shuffling

    # Save images before applying extract_features (ensure they're in 2D format here)
    if DEBUG:
        label_counts = defaultdict(int)
        for idx, (test_sample, true_label) in enumerate(zip(X_test, y_test)):
            label_counts[true_label] += 1
            count = label_counts[true_label]
            if count > 1:
                filename = f"{TEST_DIR}Character_{true_label}({count}).png"
            else:
                filename = f"{TEST_DIR}Character_{true_label}.png"
            print(f"Saving image {idx} with true label {true_label}, count {count}")
            write_image(test_sample, filename)

    # After saving images, apply extract_features (flatten images for k-NN)
    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    # Run k-NN to get predictions
    y_pred = knn(X_train, y_train, X_test, k)

    # Write images with predicted labels (after prediction)
    if DEBUG:
        for idx, (test_sample, predicted_label) in enumerate(zip(X_test, y_pred)):
            print(f"Saving image {idx} with predicted label {predicted_label}")

    # Calculate accuracy
    accuracy = sum([
        int(y_pred_i == y_test_i)
        for y_pred_i, y_test_i in zip(y_pred, y_test)
    ]) / len(y_test)
    print(f'Predicted labels: {y_pred}')
    print(f'Accuracy: {accuracy * 100}%')

    if DEBUG is not True: 
        # custom image prediction
        custom_image_path = TEST_DIR + "test.png"  # Set the custom image path to test/test.png
        custom_image = read_image(custom_image_path)  # Read the image
        custom_image_flattened = flatten_list(custom_image)  # Flatten the image for k-NN
        # prediction for the custom image
        distances = get_training_distances_for_test_sample(X_train, custom_image_flattened)
        sorted_distance_indices = [
            pair[0]
            for pair in sorted(
                enumerate(distances),
                key=lambda x: x[1]
            )
        ]
        candidates = [y_train[idx] for idx in sorted_distance_indices[:k]]
        predicted_label = get_most_frequent_element(candidates)
        print(f"The predicted label for the custom image (test.png) is: {predicted_label}")

if __name__ == '__main__':
    main()
