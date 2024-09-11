from Train import Train_VGG16, train_basic_cnn
from Misc import Image, graph, menu

import os
import csv


def tester(i):
    if i == 0:
        i = menu.menu_tester()
    false_class: int = 0
    true_class: int = 0
    models = {1: train_basic_cnn, 2: Train_VGG16}

    # Path images
    image_folder = 'media/test'

    # Path CSV
    csv_file_path = 'media/test/classes_file.csv'

    # Load the classification model (replace with your model loading function)
    model = models[i].load()

    # Load the expected classes into a dictionary from CSV
    expected_classes = {}
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_name = row['image_name']
            expected_class = row['class']
            expected_classes[image_name] = expected_class

    # Check each image in the folder
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if not image_name.lower().endswith(('jpg', 'jpeg', 'png')):
            continue  # Ignore non-image files

        # Load and process the image
        img_array = Image.load_and_preprocess_image(image_path)

        # Classify the image
        predicted_class, confidence = Image.classify_image(model, img_array)

        # Compare with expected class
        expected_class = expected_classes.get(image_name)
        if expected_class is None:
            print(f"Expected class for {image_name} not found in the CSV.")
            continue

        # Check if the forecast is correct
        if predicted_class == expected_class:
            print(f"Image {image_name} correctly classified as {predicted_class} with confidence of {confidence:.2f}.")
            true_class = true_class + 1
        else:
            print(
                f"Image {image_name} incorrectly classified. Predicted: {predicted_class}, Expected: {expected_class}.")
            false_class = false_class + 1

    graph.plot_bar(true_class, false_class)
