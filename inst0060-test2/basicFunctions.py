import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt, image as mpimg
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


seed = 4
classes = [1, 2, 3, 4, 5, 6, 7]


def train_test_split(X, y, test_ratio=0.3, seed=seed):
    """
    :param X: all features of the data
    :param y: all labels of the data
    :param test_ratio: the ratio of the testing data
    :param seed: initialize the randomly generated number
    :return: four dataframes respectively stand for features of the training data, labels of the training data,
                                                    features of the testing data, labels of the testing data.
    """
    if seed:
        np.random.seed(seed)
    shuffled_indexes = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)
    train_index = shuffled_indexes[test_size:]
    test_index = shuffled_indexes[:test_size]
    return X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]


def check_parameters(gridPara):
    """
    :param gridPara: the dictionary of parameters of the model
    :return: divided lists of different parameters
    """
    para_len = []  # the number of each parameter
    paras = []  # all parameters
    for key in gridPara.keys():
        para_len.append(len(gridPara.get(key)))
        for key_num in range(len(gridPara.get(key))):
            paras.append(gridPara.get(key)[key_num])
    # check the number of parameters
    if len(para_len) == 1:
        firstPara = paras[0:para_len[0]]
        return firstPara
    elif len(para_len) == 2:
        firstPara = paras[0:para_len[0]]
        secPara = paras[para_len[0]:(para_len[0] + para_len[1])]
        return firstPara, secPara
    elif len(para_len) == 3:
        firstPara = paras[0:para_len[0]]
        secPara = paras[para_len[0]:(para_len[0] + para_len[1])]
        thirdPara = paras[(para_len[0] + para_len[1]):sum(para_len)]
        return firstPara, secPara, thirdPara


def cross_validation(f1_overall, folds, input, model, count):
    """
    :param f1_overall: all f1-scores of the model
    :param folds: the number of the fold of Cross Validation
    :param input: Pandas Dataframe file with data
    :param model: the model (including LR, RF, SVM, KNN) using this function
    :param count: the overall running rounds
    :return: all f1-scores computed by the Cross Validation
    """
    df1 = input.loc[:, input.columns != 'Cover_Type']
    df2 = input['Cover_Type']
    f1_scores = []
    total_data_points = input.shape[0]
    for i in range(folds):
        test_data = df1.iloc[(i * int(total_data_points / folds)):((i + 1) * int((total_data_points / folds)))]
        test_targets = df2.iloc[(i * int(total_data_points / folds)):((i + 1) * int((total_data_points / folds)))]
        if i == 0:
            train_data = df1.iloc[((i + 1) * int((total_data_points / folds))):]
            train_targets = df2.iloc[((i + 1) * int((total_data_points / folds))):]
        elif i == folds - 1:
            train_data = df1.iloc[:(i * int(total_data_points / folds))]
            train_targets = df2.iloc[:(i * int(total_data_points / folds))]
        else:
            train_data_start = df1.iloc[:int((i * total_data_points / folds))]
            train_data_end = df1.iloc[int(((i + 1) * int((total_data_points / folds)))):]
            train_data = pd.concat([train_data_start, train_data_end])
            train_targets_start = df2.iloc[:int((i * total_data_points / folds))]
            train_targets_end = df2.iloc[int((i + 1) * int((total_data_points / folds))):]
            train_targets = pd.concat([train_targets_start, train_targets_end])
        model.fit(train_data, train_targets)
        test_predict = model.predict(test_data)
        f1_scores.append(f1_score(test_targets, test_predict, average='macro'))
    count += 1
    f1_overall.append(sum(f1_scores) / folds)
    print(f1_scores)
    print('Mean f1-score = ', np.mean(f1_scores))
    return f1_overall


def draw_heatmap(model, model_confusion, target, result):
    """
    :param model: model name
    :param model_confusion: the trained model
    :param target: the correct labels for test data
    :param result: the predicted labels for the test data
    :return: the corresponding heat-map and save it in fold "images"
    """
    model_confusion = confusion_matrix(target, result, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=model_confusion,
                                  display_labels=classes)
    disp.plot()
    plt.title(model, size=15, loc='center')
    plt.savefig("images/"+model+"_heatmap.jpg")
    plt.show()


def draw_comparison(all_f1_score):
    """
    :param four_f1_score: previously saved macro f1-score of the four models
    :return: the bar chart of showing the four f1-score
    """
    colors = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759"]
    x = ['LR', 'RF', 'SVM', 'KNN']
    y = all_f1_score
    plt.xticks(fontname="Times New Roman", fontsize=15)
    plt.yticks(fontname="Times New Roman", fontsize=15)
    plt.ylim(0, 0.8)
    pl = plt.bar(x, y, color=colors, width=0.5, label='value')
    plt.bar_label(pl, label_type='edge')
    plt.title("Comparison of four methods", fontsize=15)
    plt.ylabel("Macro f1 score", fontsize=15)
    plt.savefig("images/compare_f1_score.jpg")
    plt.show()


def draw_compare_heatmap():
    """
    :return: combind the four heatmaps in one picture
    """
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    image1 = mpimg.imread("images/LR_heatmap.jpg")
    plt.imshow(image1)
    plt.axis('off')
    plt.subplot(2, 2, 2)
    image2 = mpimg.imread("images/RF_heatmap.jpg")
    plt.imshow(image2)
    plt.axis('off')
    plt.subplot(2, 2, 3)
    image3 = mpimg.imread("images/SVM_heatmap.jpg")
    plt.imshow(image3)
    plt.axis('off')
    plt.subplot(2, 2, 4)
    image4 = mpimg.imread("images/KNN_heatmap.jpg")
    plt.imshow(image4)
    plt.axis('off')
    plt.tight_layout(pad=0, h_pad=-10, w_pad=-10, rect=None)
    plt.savefig("images/compare_heatmap.jpg")
    plt.show()
