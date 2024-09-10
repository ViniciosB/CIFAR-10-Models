import Misc.Image
from Misc import menu, Image
from Train import Train_VGG16
from Train import train_basic_cnn
import os
import csv
from Misc import tester
from PIL import Image
import numpy as np
true_class=0
false_class=0
#while(True):
#    operation = menu.menu_train()

 #   if operation == 1:
 #       L,S = menu.menu()
 #       model = train_basic_cnn.traincnn(int(input('Number Of Batch Size: ')), int(input('Number Of Epoch: ')), L, S)

 #   if operation == 2:
 #       L, S = menu.menu()
 #       model = Train_VGG16.trainvgg16(int(input('Number Of Batch Size: ')), int(input('Number Of Epoch: ')), L, S)


    #model = tr.train(128, 100, False, True)
    #model = Train_VGG16.load()

    #image_path = 'imagem.jpg'
    #img_array = Image.load_and_preprocess_image(image_path)
    #predicted_class, confidence = Image.classify_image(model, img_array)
    #graph.matrix(model)
    #print(f"A imagem pertence à classe: {predicted_class} com confiança de {confidence:.2f}")

tester.tester()
