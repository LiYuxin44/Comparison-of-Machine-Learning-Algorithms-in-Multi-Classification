import argparse
import time

import pandas as pd
# Data Processing
import dataProcessing
import basicFunctions
import models

# The original and globally-used data
data = pd.read_csv('data/normalized_data.csv')
input1 = pd.read_csv('data/raw_data.csv')
input2 = pd.read_csv('data/40000_data.csv')
row, col = data.shape
feature = data.iloc[:, :col - 1]
label = data.iloc[:, -1]
# random seed
seed = 4
# Class types
classes = [1, 2, 3, 4, 5, 6, 7]
# Split the processed data into train and test
X_train, X_test, y_train, y_test = basicFunctions.train_test_split(feature, label, seed=seed)
# concat the training data
train_data = pd.concat([X_train, y_train], axis=1)
c_name = train_data.columns[-1]  # last column's name
train_data.rename(columns={c_name: 'Cover_Type'}, inplace=True)
# folds of cross validation
folds = 10
# final f1-score
lr_score, rf_score, svm_score, knn_score = 0.0, 0.0, 0.0, 0.0


if __name__ == '__main__':
    # Type in to run different functions separately
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', required=False)
    args = parser.parse_args()
    if args.function:
        if args.function.__eq__("preprocess"):
            dataProcessing.preprocessing(input1)
        elif args.function.__eq__('normalization'):
            dataProcessing.normalization(input2)
        elif args.function.__eq__('LR'):
            models.logistic_regression(train_data)
        elif args.function.__eq__('RF'):
            models.random_forest(train_data)
        elif args.function.__eq__('SVM'):
            models.svm(train_data)
        elif args.function.__eq__('KNN'):
            models.knn(train_data)
        elif args.function.__eq__('heatmaps'):
            basicFunctions.draw_compare_heatmap()
        else:
            print("Please enter the right keyword.")
    # Run the complete procedure of processing data and training models
    else:
        start_time = time.time()
        print("---------------------\n"
              "Training four models sequentially.\n"
              "---------------------\n")
        rf_score = models.random_forest(train_data)
        lr_score = models.logistic_regression(train_data)
        svm_score = models.svm(train_data)
        knn_score = models.knn(train_data)
        four_score = [lr_score, rf_score, svm_score, knn_score]
        basicFunctions.draw_comparison(four_score)
        basicFunctions.draw_compare_heatmap()
        end_time = time.time()
        total_time = end_time - start_time
        print("---------------------\n"
              f"The total running time is {int(total_time/60)} minutes "
              f"{int(10*(total_time-int(60*(total_time/60))))} seconds.\n "
              f"--------------------")
