import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def preprocessing(data):
    """
    :param data: the raw data needed to be preprocessed
    :output: all features of data and the histogram of feature distribution
    :outcome: save the preprocessed data "40000_data.csv" into folder "data"
    """
    # print("---------------------\n"
    #       "Preprocessing the data now.\n"
    #       "---------------------\n")
    # col_name = data.columns[:-1]
    # label_name = data.columns[-1]
    # print('Labels of dataset：{}'.format(label_name))
    # # Check the features of the dataset
    # print('Features of dataset：{}'.format(col_name))
    # # Check the shape of the dataset
    # print('Shape of the original dataset：{}'.format(data.shape))
    # #
    # y0 = []
    # y_shape = []
    # total_rows = data.shape[0]
    # # extract classes into different elements of list
    # for i in range(1, 8):
    #     y_i = data.loc[(data['Cover_Type'] == i)]
    #     y_i = y_i.to_numpy()
    #     y0.append(y_i)
    #     y_shape.append(y_i.shape[0])
    # y_shape_ratios = np.array(y_shape) / total_rows
    # # shuffle each class and extract
    # for i in range(7):
    #     np.random.shuffle(y0[i])
    #     upper = int(np.rint(y_shape_ratios[i] * 40000))
    #     y0[i] = y0[i][:upper]
    # data_new = pd.DataFrame()
    # # concatenate into one Data Frame
    # for i in range(7):
    #     con = pd.DataFrame(y0[i])
    #     data_new = pd.concat([con, data_new])
    # data_new.columns = data.columns.values
    # # column 'Soil_Type15' full of 0, therefore we delete it.
    # data_new = data_new.drop(['Soil_Type15'], axis=1)
    # print('Shape of the new dataset：{}\n'.format(data_new.shape))
    # data_new.to_csv('data/40000_data.csv', index=False)
    # draw the histogram of feature distribution
    # i = 1
    # fig = plt.figure(figsize=(50, 50))
    # for feature in data_new.columns[:55]:
    #     axe = fig.add_subplot(24, 4, i)
    #     axe.hist(data_new[feature], bins=20)
    #     plt.title(feature)
    #     plt.savefig("images/feature_distribution.jpg")
    #     i += 1
    # plot the histogram distribution of the label "Cover_Type"
    colors = ['salmon', 'darkorange', 'pink', 'turquoise', 'skyblue', 'lightgreen', 'khaki']
    labels = ['Class1-Spruce/Fir', 'Class2-Lodgepole Pine', 'Class3-Ponderosa Pine', 'Class4-Cottonwood/Willow',
              'Class5-Aspen', 'Class6-Douglas-fir', 'Class7-Krummholz']
    x = ['1', '2', '3', '4', '5', '6', '7']
    yseries = (data['Cover_Type'].value_counts())
    y = yseries.sort_index()
    plt.figure()
    plt.bar(x, y, color=colors, label=labels)
    plt.legend(loc='upper right')
    plt.title('Cover_Type')
    plt.savefig("images/label_distribution.jpg")





def normalization(input2):
    """
    :param input2: the preprocessed data needed to be normalized
    :output: print the sentence to show the normalization is completed
    :outcome: save the normalized data "normalized_data.csv" into folder "data"
    """
    print("---------------------\n"
          "Normalizing the data now.\n"
          "---------------------\n")
    df2 = pd.DataFrame()
    df1 = input2.loc[:, input2.columns != 'Cover_Type']
    columns = df1.columns.values
    for i in columns:
        in_numpy = df1[i].to_numpy()
        rows = in_numpy.shape[0]
        in_numpy = in_numpy.reshape((-1, rows))
        minimum = np.amin(in_numpy)
        maximum = np.amax(in_numpy)
        if maximum == minimum:
            in_numpy = np.full((1, rows), minimum)
            in_numpy = in_numpy.reshape((rows, -1))
            df2 = pd.concat([df2, pd.DataFrame(in_numpy)], axis=1)
            continue
        in_numpy = (in_numpy - np.full((1, rows), minimum)) / (maximum - minimum)
        in_numpy = in_numpy.reshape((rows, -1))
        df2 = pd.concat([df2, pd.DataFrame(in_numpy)], axis=1)
    df3 = input2['Cover_Type']
    df2 = pd.concat([df2, df3], axis=1)
    df2.columns = input2.columns.values
    normalized_data = df2
    normalized_data = normalized_data.sample(frac=1).reset_index(drop=True)
    normalized_data.to_csv('data/normalized_data.csv', index=False)
    print(f"The normalized data has been save to data/normalized_data.csv.\n")


