import matplotlib.pyplot as plt


def plot(model, history, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f'Test accuracy: {test_acc}')
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()


def matrix(model):
    from tensorflow.keras import datasets
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    # Carregar o conjunto de dados CIFAR-10
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalizar os valores dos pixels entre 0 e 1
    test_images = test_images / 255.0

    # Carregar o modelo treinado
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    # Fazer predições no conjunto de teste
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)

    # Calcular a matriz de confusão
    conf_matrix = confusion_matrix(test_labels, predicted_classes)

    # Plotar a matriz de confusão usando seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Classe Predita')
    plt.ylabel('Classe Real')
    plt.title('Matriz de Confusão - CIFAR-10')
    plt.show()


def plot_bar(true_class, false_class):
    # Dados para o gráfico
    labels = ['True', 'False']
    values = [true_class, false_class]

    # Gerando o gráfico de barras
    plt.bar(labels, values, color=['green', 'red'])

    # Adicionando título e rótulos
    plt.title('True vs False Classification')
    plt.xlabel('Classification')
    plt.ylabel('Count')

    # Exibir o gráfico
    plt.show()
