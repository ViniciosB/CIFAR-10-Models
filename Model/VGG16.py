from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dropout

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import optimizers

classes = 10
vgg16_model = VGG16(weights='imagenet', include_top=False, classes=classes, input_shape=(32, 32, 3))
model = models.Sequential()

for layer in vgg16_model.layers:
    model.add(layer)

model.add(layers.Flatten())
model.add(Dropout(0.4))
model.add(layers.BatchNormalization())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(classes, activation='softmax'))
model.compile(optimizer=optimizers.RMSprop(learning_rate=2e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


def model_vgg16():
    return model
