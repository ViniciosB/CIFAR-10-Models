def menu_train():
    print('1: Train Basic CNN')
    print('2: Train VGG16')
    print('0: Exit')
    op = int(input('Input an option: '))
    if op == 0: exit()
    return op


def menu_save():
    print('1: Save Model')
    print('2: Not Save Model')
    print('0: Exit')
    op = int(input('Input an option: '))
    if op == 0: exit()
    return op


def menu_load():
    print('1: Load Model')
    print('2: Not Load Model')
    print('0: Exit')
    op = int(input('Input an option: '))
    if op == 0: exit(0)
    return op

def menu():
    operation_load = True if menu_load() == 1 else False
    operation_save = True if menu_save() == 1 else False
    return operation_load, operation_save


def menu_initial():
    print('1: Test')
    print('2: Train')
    print('3: Confusion Matrix')
    print('0: Exit')
    op = int(input('Input an option: '))
    if op == 0: exit()
    return op


def menu_tester():
    print('1: Test Basic CNN')
    print('2: Test VGG16')
    print('0: Exit')
    op = int(input('Input an option: '))
    if op == 0: exit()
    return op
def menu_matrix():
    print('1: Confusion Matrix Basic CNN')
    print('2: Confusion Matrix Test VGG16')
    print('0: Exit')
    op = int(input('Input an option: '))
    if op == 0: exit()
    return op