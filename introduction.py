import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# load the exel file
df = pd.read_csv("train.csv")

# array of all strings of any pozx
all_X = []
for i in range(0, 30):
    all_X.append("posX_" + str(i))
all_X.append("class")

# array of all strings of any pozz
all_Y = []
for i in range(0, 30):
    all_Y.append("posZ_" + str(i))
all_Y.append("class")


def question1():
    global df
    # use the first column as index and delete the column "targetName"
    df.rename(columns={df.columns[0]: "index_table"}, inplace=True)
    df = df.set_index('index_table')
    df.drop(['targetName'], axis=1, inplace=True)

    # show number of paths for each class type
    num_class_type = df['class'].value_counts()

    # showing a histogram of the paths length of each type
    all_times = []
    for i in range(1, 30):
        all_times.append("Time_" + str(i))

    for i in range(1, max(df["class"]) + 1):
        times = df.loc[df['class'] == i, all_times]
        max_times = times.max(axis=1)

        max_times.plot.hist(bins=50, alpha=0.75)
        plt.show()


"""draws the path of the first rocket"""


def question2_a():
    global df
    x_list = df.loc[0, all_X]
    x_list = np.array(x_list)

    y_list = df.loc[0, all_Y]
    y_list = np.array(y_list)
    plt.plot(x_list, y_list)
    plt.show()


""""draws the first 50 rocket path for each type a different colour"""
def question2_b():
    global df, all_X, all_Y
    colors = ["red", "green", "blue", "yellow", "brown", "pink"]
    x_list = df.loc[df["class"] <= 6, all_X]
    x_list = x_list.head(50)
    y_list = df.loc[df["class"] <= 6, all_Y]
    y_list = y_list.head(50)
    for i in range(1, 7):
        current_x = x_list.loc[x_list["class"] == i, x_list.columns != "class"]
        current_y = y_list.loc[y_list["class"] == i, y_list.columns != "class"]
        for j in range(len(current_x)):
            plt.plot(np.array(current_x)[j], np.array(current_y)[j], color=colors[i - 1])

    plt.show()

""""draws the first 50 rocket path that take 15 seconds from 1 and 6 type"""
def quiation2_c():
    global df, all_X, all_Y
    x_list = df.loc[((df["class"] == 1) | (df["class"] == 6)) & (df["Time_29"] == 14.5), all_X]
    x_list = x_list.head(50)
    y_list = df.loc[((df["class"] == 1) | (df["class"] == 6)) & (df["Time_29"] == 14.5), all_Y]
    y_list = y_list.head(50)
    for i in range(1, 7):
        current_x = x_list.loc[x_list["class"] == i, x_list.columns != "class"]
        current_y = y_list.loc[y_list["class"] == i, y_list.columns != "class"]
        for j in range(len(current_x)):
            plt.plot(np.array(current_x)[j], np.array(current_y)[j])

    plt.show()
