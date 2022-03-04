# CSE4088 - Introduction to Machine Learning
# Term Project
# Part 2
# 150119824 - Zeynep Ferah Akkurt
# 150119825 - Merve Rana Kızıl
# DatasetPlots.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DatasetPlot:
    def __init__(self, MushroomData):
        self.df = MushroomData

    # saving the plot in feature_figures folder
    def savePlot(self, feature):
        mainPath = os.getcwd()
        folderName = mainPath + '/feature_figures'
        if not os.path.exists(folderName):
            os.makedirs(folderName)
        filename = folderName + '/' + str(feature) + ".png"

        plt.savefig(filename, dpi=300)

    # plotting one feature
    def plotDataFeatures(self, feature):
        a = plt.figure()
        # plot the dataframe
        plt.xlabel(feature)
        plt.ylabel("Number of instances")
        title = "Distribution of feature values"
        plt.title(title)

        self.df[feature].value_counts(normalize=False, ascending=False).plot(kind='bar', color=plt.cm.Paired(
            np.arange(len(feature))), rot=0)

        plt.tight_layout()
        self.savePlot(feature)
        plt.close(a)

    # general look on dataset and gives an output as datasetInfo.txt
    def examineDataset(self):

        with open('datasetInfo.txt', 'w') as f:
            f.write("General look on dataset: \n")
            f.write(str(self.df.head(5)))
            m = '\n' + 'Number of rows:' + str(self.df.shape[0]) + 'Number of columns: ' + str(self.df.shape[1]) + '\n'
            f.write(m)
            # number of null values for each column
            m2 = str(self.df.isnull().values.any()) + ":There is no null value" + '\n'
            f.write(m2)
            # total columns description
            f.write(str(self.df.describe().T))

            fno = 0
            for feature in self.df.columns:
                m = '\n' + "------------" + str(feature) + "------------" + '\n' + "Feature number:" + str(fno + 1)
                f.write(m)
                # number of classes
                f.write("\nunique feature values: ")
                f.write(str(self.df[feature].unique()))
                f.write("\nPercentage of feature values: \n")
                values = self.df[feature].value_counts(normalize=True, ascending=True)
                f.write(str(values))
                f.write("\n")

                self.plotDataFeatures(feature)
                fno += 1

    # plots all feature as one figure for general stats, one for based on their class
    def plotAll(self):
        p, axes = plt.subplots(nrows=4, ncols=6, figsize=(6, 4))
        p2, axes2 = plt.subplots(nrows=4, ncols=6, figsize=(6, 4))
        p.tight_layout()
        p.suptitle('Plots for Categorical Variables of Classes', fontsize=15)
        p2.tight_layout()
        p2.suptitle('Plots for Categorical Variables', fontsize=15)

        x_axes = 0
        y_axes = 0
        color = ['red', 'lightgreen']

        fno = 0
        for feature in self.df.columns:
            a = sns.countplot(data=self.df, x=feature, hue='class', palette=color, ax=axes[x_axes, y_axes])
            a.legend([], [], frameon=False)
            sns.countplot(data=self.df, x=feature, ax=axes2[x_axes, y_axes])

            x_axes = int(fno / 6)
            y_axes = int(fno % 6)
            fno += 1

        p.legend(['p', 'e'], loc='upper right')
        plt.show()

    # plots correlation matrix of features
    def correlationMatrix(self, df):
        with open('datasetInfo.txt', 'a') as f:
            f.write("\nCorrelation:\n")
            f.write(str(df.head()))
            f.write(str(df.corr().T[:1]))

            corrPlot = plt.figure(figsize=(12, 10))
            sns.heatmap(df.corr())
            self.savePlot("correlationMatrix")
            plt.close(corrPlot)

