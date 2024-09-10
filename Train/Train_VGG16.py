from Model import VGG16
from Misc import graph
import tensorflow as tf
from tensorflow.keras import datasets


def trainvgg16(BS, EPC, LOAD, SAVE):
    model = VGG16.model_vgg16()
    if LOAD:
        model = tf.keras.models.load_model('Model/saves/Cifar10_VGG16.h5')

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    history = model.fit(train_images, train_labels, epochs=EPC, validation_data=(test_images, test_labels),
                        batch_size=BS)

    graph.plot(model, history, test_images, test_labels)

    if SAVE:
        model.save('Model/saves/Cifar10_VGG16.h5')

    return model


def load():
    return tf.keras.models.load_model('Model/saves/Cifar10_VGG16.h5')
