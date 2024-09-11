# CIFAR-10 

## Description

The **CIFAR-10** (Canadian Institute for Advanced Research) dataset is one of the most widely used datasets for image classification tasks, especially in deep learning and computer vision. It was introduced by Alex Krizhevsky and Geoffrey Hinton, consisting of **60,000 color images of 32x32 pixels** spread across **10 different classes**. Each class contains **6,000 images**. CIFAR-10 is a smaller, less complex version of CIFAR-100, which contains 100 classes.

### Classes

The dataset contains the following 10 classes:
<p align="center">
  <img src="https://production-media.paperswithcode.com/datasets/4fdf2b82-2bc3-4f97-ba51-400322b228b1.png" alt="classes">
</p>

These classes are mutually exclusive, meaning there is no overlap between them. Each image is labeled with one of these categories.

### Data Structure

- **60,000 total images**
  - **50,000** training images
  - **10,000** test images
- The images are in color (RGB) and have a fixed size of 32x32 pixels.

### Data Format

The CIFAR-10 dataset is available in different formats, such as **Python**, **MATLAB**, and **Binary**. It can be easily imported into popular machine learning libraries like TensorFlow and PyTorch.

### Sample Images

Here are some sample images from the dataset:

- Airplane: ![Image Example](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane1.png)
- Dog: ![Image Example](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog4.png)

## How to Use CIFAR-10



You can download and use CIFAR-10 in various machine learning frameworks. Below is an example of how to load CIFAR-10 in **TensorFlow**:

```python
import tensorflow as tf

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize the images to a range of [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert class vectors to one-hot encoded labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build a simple model as an example
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')

```
### Applications of CIFAR-10

CIFAR-10 is widely used as a benchmark for:

- Image Classification
- Training Convolutional Neural Networks (CNN)
- Transfer Learning
- Studies in Model Regularization and Optimization

It is an excellent starting point for deep learning experiments due to its simplicity and the availability of many examples.

## References

- [Official CIFAR-10 page](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Original Paper by Krizhevsky](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
