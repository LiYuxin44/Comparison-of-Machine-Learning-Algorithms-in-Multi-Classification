import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix
import basicFunctions
import main
from main import X_train, y_train, X_test, y_test, classes, seed
import warnings
warnings.filterwarnings("ignore")


"""
    Inputs :-
        all inputs for functions in models.py is the training data
    Output :-
        each model will return the final f1-score.
        each model will output all f1-scores from each fold of the Cross-Validation,
        the final f1-score obtained from running the model on the test data
        and the heatmap of the final result of this model.
        For the model with Grid Search, it will additionally output the best parameters used in the final train and test
"""


def logistic_regression(data):
    print("Logistic Regression is running.")
    # tuning the hyperparameters
    # f1_overall = []
    # f1_final = []
    # grid_para = {"ratio": list(np.arange(0, 1.1, 0.1))}
    # para1 = basicFunctions.check_parameters(grid_para)
    # count = 0
    # for a in para1:
    #     log_reg = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=a)
    #     f1_final = basicFunctions.cross_validation(f1_overall, 10, data, log_reg, count)
    # scores = np.array(f1_final).reshape((len(para1)))
    # m = np.where(scores == np.max(scores))  # index where reaches the highest f1-score, type: tuple
    # m = m[0][0]
    # print(f"The best parameter is {para1[m]}.")

    # the best parameter for ratio is 1.0
    lr = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=1)
    # basicFunctions.cross_validation(10, data, lr, classes)
    # Final train after grid+cv
    lr.fit(X_train, y_train)
    y_test_pred_lr = lr.predict(X_test)
    lr_score = f1_score(y_test, y_test_pred_lr, average='macro')
    print("F1 score of Logistic Regression:", f1_score(y_test, y_test_pred_lr, average='macro'))
    print("confusion matrix:", confusion_matrix(y_test, y_test_pred_lr, labels=classes))
    print()
    basicFunctions.draw_heatmap("LR", lr, y_test, y_test_pred_lr)
    return lr_score


def random_forest(data):
    print("Random Forest is running.")
    # tuning the hyperparameters
    f1_overall = []
    f1_final = []
    grid_para = {'max_leaf_nodes': [1500, 2000], 'n_estimators': [90, 100], 'max_depth': [30, 40, 50]}
    para1, para2, para3 = basicFunctions.check_parameters(grid_para)
    count = 0
    for a in para1:
        for b in para2:
            for c in para3:
                forest = RandomForestClassifier(max_leaf_nodes=a,
                                                n_estimators=b,
                                                max_depth=c, n_jobs=-1, random_state=seed + 1)
                f1_final = basicFunctions.cross_validation(f1_overall, 10, data, forest, count)
    scores = np.array(f1_final).reshape((len(para1), len(para2), len(para3)))
    m, n, k = np.where(scores == np.max(scores))  # index where reaches the highest f1-score, type: tuple
    print(f"The best first parameter is {para1[int(m)]},"
          f"\n the best second parameter is {para2[int(n)]},"
          f"\n the best third parameter is {para3[int(k)]}.")
    # Best parameters are 2000, 90, 50
    forest = RandomForestClassifier(max_leaf_nodes=para1[int(m)],
                                    n_estimators=para2[int(n)],
                                    max_depth=para3[int(k)], n_jobs=-1, random_state=seed + 1)
    forest.fit(X_train, y_train)
    y_test_pred_forest = forest.predict(X_test)
    rf_score = f1_score(y_test, y_test_pred_forest, average='macro')
    print("F1 score of Random Forest:", f1_score(y_test, y_test_pred_forest, average='macro'))
    print("confusion matrix:", confusion_matrix(y_test, y_test_pred_forest, labels=classes))
    print()
    basicFunctions.draw_heatmap("RF", forest, y_test, y_test_pred_forest)
    return rf_score


def svm(data):
    print("SVM is running.")
    # tuning the hyperparameters
    # f1_overall = []
    # f1_final = []
    # grid_para = {'C': [0.1,1,10], 'gamma': [0.01,0.1,1]}
    # para1, para2 = basicFunctions.check_parameters(grid_para)
    # count = 0
    # for a in para1:
    #     for b in para2:
    #         svc = SVC(C=a, gamma=b, kernel='rbf', probability=True)
    #         f1_final = basicFunctions.cross_validation(f1_overall, 10, data, svc, count)
    # scores = np.array(f1_final).reshape((len(para1), len(para2)))
    # m, n = np.where(scores == np.max(scores))  # index where reaches the highest f1-score, type: tuple
    # print(f"The best first parameter is {para1[int(m)]},"
    #       f"\n the best second parameter is {para2[int(n)]}.")

    # Best parameters are C=10 and gamma=1.
    svc = SVC(C=10, gamma=1, kernel='rbf', probability=True)
    svc.fit(X_train, y_train)
    y_test_pred_svc = svc.predict(X_test)
    svm_score = f1_score(y_test, y_test_pred_svc, average='macro')
    print("F1 score of SVC:", f1_score(y_test, y_test_pred_svc, average='macro'))
    print("confusion matrix:", confusion_matrix(y_test, y_test_pred_svc, labels=classes))
    print()
    basicFunctions.draw_heatmap("SVM", svc, y_test, y_test_pred_svc)
    return svm_score


def knn(data):
    print("KNN is running.")
    # f1_overall = []
    # f1_final = []
    # grid_para = {"n_neighbors": list(np.arange(1, 6, 1))}
    # para1 = basicFunctions.check_parameters(grid_para)
    # count = 0
    # for a in para1:
    #     neigh = KNeighborsClassifier(n_neighbors=a)
    #     f1_final = basicFunctions.cross_validation(f1_overall, 10, data, neigh, count)
    # scores = np.array(f1_final).reshape((len(para1)))
    # m = np.where(scores == np.max(scores))  # index where reaches the highest f1-score, type: tuple
    # m = m[0][0]
    # print(f"The best parameter is {para1[m]}.")

    # The best parameter is n_neightbors=1.
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X_train, y_train)
    y_test_pred_knn = neigh.predict(X_test)
    knn_score = f1_score(y_test, y_test_pred_knn, average='macro')
    print("F1 score of KNN:", f1_score(y_test, y_test_pred_knn, average='macro'))
    print("confusion matrix:", confusion_matrix(y_test, y_test_pred_knn, labels=classes))
    print()
    basicFunctions.draw_heatmap("KNN", neigh, y_test, y_test_pred_knn)
    return knn_score
