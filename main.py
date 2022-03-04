# CSE4088 - Introduction to Machine Learning
# Term Project
# Part 1
# 150119824 - Zeynep Ferah Akkurt
# 150119825 - Merve Rana Kızıl
# main.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from DatasetPlot import *
from Algorithm import *
import os


# using a copy and changing the all dataframe as numerically, then extract X and y from it
def labelEncoder(df):
    for col in df.columns:
        le = LabelEncoder()
        le.fit(df[str(col)])
        df[str(col)] = le.transform(df[str(col)])

    X = df.iloc[:, 1:23]  # all rows, all the features and no labels
    y = df.iloc[:, 0]

    return X, y


# transformation of X and y numerically and returns directly (not changing the dataframe)
def oneHotEncoder(df):
    # categorized --> numeric values
    X = df.drop(["class"], axis=1)
    y = df["class"]

    X = pd.get_dummies(X)
    # we need to encode our y-labels numerically
    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y


def main():
    path = os.getcwd()
    fileName = path + '\mushrooms.csv'
    # load the data
    data = pd.read_csv(fileName)

    print("Examine the dataset:")
    # examine and plot the dataset
    dataplots = DatasetPlot(data)
    dataplots.examineDataset()
    dataplots.plotAll()

    # check correlation
    corr_df = dataplots.df.copy()
    labelEncoder(corr_df)
    dataplots.correlationMatrix(corr_df)

    # encode the dataset as numerically to train and test
    #X, y = labelEncoder(data)
    X, y = oneHotEncoder(data)


    # training and testing with different algorithms
    algorithm = Algorithm(X, y)
    algorithm.supportVectorMachines()
    algorithm.naiveBayes()
    algorithm.k_NearestNeighbor()
    algorithm.randomForest()
    algorithm.neuralNetwork()


if __name__ == "__main__":
    main()
