
from Misc import menu, graph
from Train import Train_VGG16
from Train import train_basic_cnn
from Misc import tester

while True:
    operation = menu.menu_initial()
    if operation == 1:
        tester.tester(0)
    if operation == 2:
        operation = menu.menu_train()

        if operation == 1:
            L, S = menu.menu()
            model = train_basic_cnn.traincnn(int(input('Number Of Batch Size: ')), int(input('Number Of Epoch: ')), L, S)
            tester.tester(1)

        if operation == 2:
            L, S = menu.menu()
            model = Train_VGG16.trainvgg16(int(input('Number Of Batch Size: ')), int(input('Number Of Epoch: ')), L, S)
            tester.tester(2)
    if operation == 3:
        operation=menu.menu_matrix()
        models = {1: train_basic_cnn, 2: Train_VGG16}
        graph.matrix(models[operation].load())