# CSE4088 - Introduction to Machine Learning
# Term Project
# Part 3
# 150119824 - Zeynep Ferah Akkurt
# 150119825 - Merve Rana Kızıl
# Algorithms.py

import os

import matplotlib.pyplot as plt
import seaborn
import tensorflow as tf
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

tensorflow_version = float(tf.__version__[0:3])
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Algorithm:
    def __init__(self, X, y):
        self.X = X
        self.y = y

        # split the dataset for %80 training and %20 testing
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2,shuffle=True)

        self.y_pred = 0
        self.classification_method_str = ""

        # accuracies of each algorithm
        self.svm_acc = 0
        self.nb_Acc = 0
        self.knn_acc = 0
        self.rf_acc = 0
        self.ann_acc = 0
        self.ann_loss = 0

    def supportVectorMachines(self):
        self.classification_method_str = "SVM"

        svm = SVC(random_state=45, gamma="auto")
        svm.fit(self.x_train, self.y_train)
        self.svm_acc = round(svm.score(self.x_test, self.y_test) * 100, 2)
        self.y_pred = svm.predict(self.x_test)

        print("Test Accuracy of SVM: {}%".format(self.svm_acc))

        self.plot_confusion_matrix()

    def naiveBayes(self):
        self.classification_method_str = "Naive Bayes"
        nb = GaussianNB()
        nb.fit(self.x_train, self.y_train)
        self.nb_Acc = round(nb.score(self.x_test, self.y_test) * 100, 2)
        self.y_pred = nb.predict(self.x_test)

        print("Test Accuracy of NB: {}%".format(self.nb_Acc))
        self.plot_confusion_matrix()

    def k_NearestNeighbor(self):
        self.classification_method_str = "KNN"
        knn = KNeighborsClassifier()
        knn_model = knn.fit(self.x_train, self.y_train)
        # knn_model
        self.y_pred = knn_model.predict(self.x_test)
        self.knn_acc = round(metrics.accuracy_score(self.y_test, self.y_pred) * 100, 2)
        print("Test Accuracy of kNN: {}%".format(self.knn_acc))
        self.plot_confusion_matrix()

    def randomForest(self):
        self.classification_method_str = "Random Forest"
        rf_model = RandomForestClassifier().fit(self.x_train, self.y_train)
        self.y_pred = rf_model.predict(self.x_test)
        self.rf_acc = round(metrics.accuracy_score(self.y_test, self.y_pred) * 100, 2)
        print("Test Accuracy of random forest: {}%".format(self.rf_acc))
        self.plot_confusion_matrix()

    def neuralNetwork(self):
        self.classification_method_str = "ANN"
        # split training set as training and validation
        X_train, X_val, y_train, y_val = train_test_split(self.x_train, self.y_train, test_size=0.2, shuffle=True)

        # model
        model = Sequential([
            layers.Dense(32, input_dim=X_train.shape[1], activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid'),
        ])

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        # Train the model
        # early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)
        epochs = 15
        history = model.fit(x=X_train,
                            y=y_train,
                            epochs=epochs,
                            batch_size=200,
                            validation_data=(X_val, y_val),
                            # callbacks=[early_stop]
                            )

        # results of training
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        # plotting training vs validation for accuracy
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.title('Training and Validation Accuracy')

        # plotting training vs validation for loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.title('Training and Validation Loss')
        plt.savefig(os.getcwd() + '/training&validation.png')

        # Test the model
        self.ann_loss, self.ann_acc = model.evaluate(self.x_test, self.y_test)
        self.ann_loss, self.ann_acc = self.ann_loss * 10, self.ann_acc * 100
        print("Test Accuracy of ANN: {}%".format(self.ann_acc), ", Loss of ANN: {}%".format(self.ann_loss))

        # save the model
        model.save("model artifact")

    # plotting confusion matrix for the algorithm
    def plot_confusion_matrix(self):
        conf_matrix = confusion_matrix(self.y_test, self.y_pred)

        x_axis_labels = ["Edible", "Poisonous"]
        y_axis_labels = ["Edible", "Poisonous"]

        f, ax = plt.subplots(figsize=(7, 7))
        seaborn.heatmap(conf_matrix, annot=True, linewidths=0.2, linecolor="black", fmt=".0f", ax=ax, cmap="Blues",
                        xticklabels=x_axis_labels, yticklabels=y_axis_labels)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix for " + self.classification_method_str + " Classifier")
        plt.show()
