# 分类算法
# auc
# acc
# roc  =（fpr，tpr）
# 混淆矩阵
# print('precision:%.3f' % precision_score(y_true=y_test, y_pred=y_pred,average= 'macro'))
# print('recall:%.3f' % recall_score(y_true=y_test, y_pred=y_pred,average= 'macro'))
# print('F1:%.3f' % f1_score(y_true=y_test, y_pred=y_pred,average= 'macro'))
# # 搜寻到的最佳模型
# print('搜寻到的最佳模型:', rnd_search_cv_forest.best_estimator_)
# # 最佳参数
# print('最佳参数:', rnd_search_cv_forest.best_score_)
'''
# rf
dict1 = train_rf_classifier(X_train, X_val, y_train, y_val)
# svr
dict2 = train_svc(X_train, X_val, y_train, y_val)
# SGB
dict3 = train_SGD(X_train, X_val, y_train, y_val)
# xgb
dict4=train_xgboost_classifier(X_train, X_val, y_train, y_val)
# gbrt
dict5=train_GBRT_classifier(X_train, X_val, y_train, y_val)
# knn
dict6=train_KNeighborsClassifier(X_train, X_val, y_train, y_val)
# stacking
dict7=train_StackingClassifier(X_train, X_val, y_train, y_val)
#MLP
dict8=train_MLP_Classifier(X_train, X_val, y_train, y_val)
#DT
dict9=train_DecisionTreeClassifier(X_train, X_val, y_train, y_val)
#lgb
dict10=train_LGBMClassifier(X_train, X_val, y_train, y_val)
#bagging
dict11=train_BaggingClassifier(X_train, X_val, y_train, y_val)
'''



import multiprocessing
import time

import numpy as np
import xgboost
from matplotlib import pyplot as plt
from scipy.stats import randint, reciprocal, uniform
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, BaggingClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, train_test_split, GridSearchCV
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from sklearn.metrics import confusion_matrix



def train_rf_classifier(X_train, X_val, y_train, y_val,name):
    start = time.perf_counter()
    param_distributions = {'n_estimators': randint(low=1, high=200), 'max_features': randint(low=1, high=8), }
    forest_clf = RandomForestClassifier(random_state=42)
    clf = RandomizedSearchCV(forest_clf, param_distributions, n_iter=100, verbose=2, cv=5, n_jobs=16)
    clf.fit(X_train, y_train)
    end = time.perf_counter()
    print('rfc_best Running time: %s Seconds' % (end - start))
    joblib.dump(clf.best_estimator_, name) # 'rfc_best.pkl'
    dic = {}
    dic['accuracy_train'] = sum(cross_val_score(clf.best_estimator_, X_train, y_train, cv=10, scoring="accuracy")) / 10
    dic['accuracy_val'] = sum(cross_val_score(clf.best_estimator_, X_val, y_val, cv=10, scoring="accuracy")) / 10
    y_probas_val = cross_val_predict(clf.best_estimator_, X_val, y_val, cv=5, method="predict_proba")
    y_scores_val = y_probas_val[:, 1]  # score = proba of positive class
    y_probas_train = cross_val_predict(clf.best_estimator_, X_train, y_train, cv=5, method="predict_proba")
    y_scores_train = y_probas_train[:, 1]
    dic['AUC_train'] = roc_auc_score(y_train, y_scores_train)
    dic['AUC_val'] = roc_auc_score(y_val, y_scores_val)
    fpr, tpr, thresholds = roc_curve(y_val, y_scores_val)
    dic['fpr'] = fpr
    dic['tpr'] = tpr
    dic_data = {}
    dic_data['y_val'] = list(y_val)
    dic_data['y_pred'] = list(clf.best_estimator_.predict(X_val))
    return dic, dic_data


def train_svc(X_train, X_val, y_train, y_val):
    start = time.perf_counter()
    svm_clf = SVC(decision_function_shape="ovr", gamma="auto", probability=True)
    param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
    clf = RandomizedSearchCV(svm_clf, param_distributions, n_iter=10, verbose=2, cv=3, n_jobs=1)
    clf.fit(X_train, y_train)
    end = time.perf_counter()
    print('svc_best Running time: %s Seconds' % (end - start))
    joblib.dump(clf.best_estimator_, 'svc_best.pkl')
    dic = {}
    dic['accuracy_train'] = sum(cross_val_score(clf.best_estimator_, X_train, y_train, cv=10, scoring="accuracy")) / 10
    dic['accuracy_val'] = sum(cross_val_score(clf.best_estimator_, X_val, y_val, cv=10, scoring="accuracy")) / 10
    y_probas_val = cross_val_predict(clf.best_estimator_, X_val, y_val, cv=5, method="predict_proba")
    y_scores_val = y_probas_val[:, 1]  # score = proba of positive class
    y_probas_train = cross_val_predict(clf.best_estimator_, X_train, y_train, cv=5, method="predict_proba")
    y_scores_train = y_probas_train[:, 1]
    dic['AUC_train'] = roc_auc_score(y_train, y_scores_train)
    dic['AUC_val'] = roc_auc_score(y_val, y_scores_val)
    fpr, tpr, thresholds = roc_curve(y_val, y_scores_val)
    dic['fpr'] = fpr
    dic['tpr'] = tpr
    dic_data = {}
    dic_data['y_val'] = list(y_val)
    dic_data['y_pred'] = list(clf.best_estimator_.predict(list(X_val)))
    return dic, dic_data


def train_SGD(X_train, X_val, y_train, y_val):
    start = time.perf_counter()
    clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42, n_jobs=200)
    clf.fit(X_train, y_train)
    end = time.perf_counter()
    print('sgd_best Running time: %s Seconds' % (end - start))
    joblib.dump(clf, 'sgdc_best.pkl')
    dic = {}
    dic['accuracy_train'] = sum(cross_val_score(clf, X_train, y_train, cv=10, scoring="accuracy")) / 10
    dic['accuracy_val'] = sum(cross_val_score(clf, X_val, y_val, cv=10, scoring="accuracy")) / 10
    y_scores_val = cross_val_predict(clf, X_val, y_val, cv=5, method="decision_function")
    y_scores_train = cross_val_predict(clf, X_train, y_train, cv=5, method="decision_function")
    dic['AUC_train'] = roc_auc_score(y_train, y_scores_train)
    dic['AUC_val'] = roc_auc_score(y_val, y_scores_val)
    fpr, tpr, thresholds = roc_curve(y_val, y_scores_val)

    dic['fpr'] = fpr
    dic['tpr'] = tpr
    dic_data ={}
    dic_data['y_val']=list(y_val)
    dic_data['y_pred']=list(clf.predict(X_val))
    return dic, dic_data


def train_xgboost_classifier(X_train, X_val, y_train, y_val):
    # xgboost.XGBRFClassifier
    start = time.perf_counter()
    xgb_model = xgboost.XGBClassifier(n_jobs=multiprocessing.cpu_count() // 2)
    clf = GridSearchCV(xgb_model, {'max_depth': [2, 4, 6],
                                   'n_estimators': [50, 100, 200]}, verbose=1, n_jobs=2)
    clf.fit(X_train, y_train)
    end = time.perf_counter()
    print('xgbc_best Running time: %s Seconds' % (end - start))
    joblib.dump(clf.best_estimator_, 'xgbc_best.pkl')
    dic = {}
    dic['accuracy_train'] = sum(cross_val_score(clf.best_estimator_, X_train, y_train, cv=10, scoring="accuracy")) / 10
    dic['accuracy_val'] = sum(cross_val_score(clf.best_estimator_, X_val, y_val, cv=10, scoring="accuracy")) / 10
    y_probas_val = cross_val_predict(clf.best_estimator_, X_val, y_val, cv=5, method="predict_proba")
    y_scores_val = y_probas_val[:, 1]  # score = proba of positive class
    y_probas_train = cross_val_predict(clf.best_estimator_, X_train, y_train, cv=5, method="predict_proba")
    y_scores_train = y_probas_train[:, 1]
    dic['AUC_train'] = roc_auc_score(y_train, y_scores_train)
    dic['AUC_val'] = roc_auc_score(y_val, y_scores_val)
    fpr, tpr, thresholds = roc_curve(y_val, y_scores_val)
    dic['fpr'] = fpr
    dic['tpr'] = tpr
    dic_data = {}
    dic_data['y_val'] = list(y_val)
    dic_data['y_pred'] = list(clf.best_estimator_.predict(X_val))
    return dic, dic_data


# 10min
def train_GBRT_classifier(X_train, X_val, y_train, y_val):
    start = time.perf_counter()
    # HistGradientBoostingRegressor 有时间研究下
    gbrt = GradientBoostingClassifier(random_state=42, learning_rate=0.1)
    param_distribs = {'max_depth': randint(low=2, high=20), 'n_estimators': randint(low=20, high=200), }
    clf = RandomizedSearchCV(gbrt, param_distributions=param_distribs, n_iter=100, cv=5, scoring='r2',
                             random_state=42)
    clf.fit(X_train, y_train)
    end = time.perf_counter()
    print('gbrtc_best Running time: %s Seconds' % (end - start))
    joblib.dump(clf.best_estimator_, 'gbrtc_best.pkl')
    dic = {}
    dic['accuracy_train'] = sum(cross_val_score(clf.best_estimator_, X_train, y_train, cv=10, scoring="accuracy")) / 10
    dic['accuracy_val'] = sum(cross_val_score(clf.best_estimator_, X_val, y_val, cv=10, scoring="accuracy")) / 10
    y_probas_val = cross_val_predict(clf.best_estimator_, X_val, y_val, cv=5, method="predict_proba")
    y_scores_val = y_probas_val[:, 1]  # score = proba of positive class
    y_probas_train = cross_val_predict(clf.best_estimator_, X_train, y_train, cv=5, method="predict_proba")
    y_scores_train = y_probas_train[:, 1]
    dic['AUC_train'] = roc_auc_score(y_train, y_scores_train)
    dic['AUC_val'] = roc_auc_score(y_val, y_scores_val)
    fpr, tpr, thresholds = roc_curve(y_val, y_scores_val)
    dic['fpr'] = fpr
    dic['tpr'] = tpr
    dic_data = {}
    dic_data['y_val'] = list(y_val)
    dic_data['y_pred'] = list(clf.best_estimator_.predict(X_val))
    return dic, dic_data


def train_KNeighborsClassifier(X_train, X_val, y_train, y_val):
    start = time.perf_counter()
    model = KNeighborsClassifier(n_jobs=-1)
    # 尝试使用网格搜索优化
    param_grid = [{'weights': ['uniform'],
                   'n_neighbors': [k for k in range(1, 8)]
                   },
                  {'weights': ['distance'],
                   'n_neighbors': [k for k in range(1, 8)],
                   'p': [p for p in range(1, 8)]}]
    clf = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='r2', verbose=1)
    clf.fit(X_train, y_train)
    end = time.perf_counter()
    print('knnc Running time: %s Seconds' % (end - start))
    joblib.dump(clf.best_estimator_, 'knnc_best.pkl')
    dic = {}
    dic['accuracy_train'] = sum(cross_val_score(clf.best_estimator_, X_train, y_train, cv=10, scoring="accuracy")) / 10
    dic['accuracy_val'] = sum(cross_val_score(clf.best_estimator_, X_val, y_val, cv=10, scoring="accuracy")) / 10
    y_probas_val = cross_val_predict(clf.best_estimator_, X_val, y_val, cv=5, method="predict_proba")
    y_scores_val = y_probas_val[:, 1]  # score = proba of positive class
    y_probas_train = cross_val_predict(clf.best_estimator_, X_train, y_train, cv=5, method="predict_proba")
    y_scores_train = y_probas_train[:, 1]
    dic['AUC_train'] = roc_auc_score(y_train, y_scores_train)
    dic['AUC_val'] = roc_auc_score(y_val, y_scores_val)
    fpr, tpr, thresholds = roc_curve(y_val, y_scores_val)
    dic['fpr'] = fpr
    dic['tpr'] = tpr
    dic_data = {}
    dic_data['y_val'] = list(y_val)
    dic_data['y_pred'] = list(clf.best_estimator_.predict(X_val))
    return dic, dic_data


# 8min
def train_StackingClassifier(X_train, X_val, y_train, y_val):
    start = time.perf_counter()
    estimators = [('rf', RandomForestClassifier(n_jobs=-1)),
                  ('svr', SVC(kernel="rbf")),
                  ('gdbt', GradientBoostingClassifier()),
                  ('knn', KNeighborsClassifier(n_jobs=-1))]
    clf = StackingClassifier(estimators=estimators, final_estimator=PassiveAggressiveClassifier())
    clf.fit(X_train, y_train)
    end = time.perf_counter()
    print('Stackc Running time: %s Seconds' % (end - start))
    joblib.dump(clf, 'Stackc_best.pkl')
    dic = {}
    dic['accuracy_train'] = sum(cross_val_score(clf, X_train, y_train, cv=10, scoring="accuracy")) / 10
    dic['accuracy_val'] = sum(cross_val_score(clf, X_val, y_val, cv=10, scoring="accuracy")) / 10
    y_scores_val = cross_val_predict(clf, X_val, y_val, cv=5, method="decision_function")
    y_scores_train = cross_val_predict(clf, X_train, y_train, cv=5, method="decision_function")
    dic['AUC_train'] = roc_auc_score(y_train, y_scores_train)
    dic['AUC_val'] = roc_auc_score(y_val, y_scores_val)
    fpr, tpr, thresholds = roc_curve(y_val, y_scores_val)
    dic['fpr'] = fpr
    dic['tpr'] = tpr
    dic_data = {}
    dic_data['y_val'] = list(y_val)
    dic_data['y_pred']=list(clf.predict(X_val))
    return dic, dic_data


def train_MLP_Classifier(X_train, X_val, y_train, y_val):
    start = time.perf_counter()
    parameters = {'hidden_layer_sizes': [200, 250, 300, 400, 500, 600], 'activation': ['relu']}
    model = MLPClassifier(
        solver='adam',
        # '''，激活函数用relu，梯度下降方法用adam'''
        alpha=0.01, max_iter=500)
    '''惩罚系数为0.01，最大迭代次数为200'''
    clf = GridSearchCV(model, parameters, cv=10, n_jobs=4, scoring='r2')
    clf.fit(X_train, y_train)
    end = time.perf_counter()
    print('MLPC Running time: %s Seconds' % (end - start))
    joblib.dump(clf.best_estimator_, 'MLPC_best.pkl')
    dic = {}
    dic['accuracy_train'] = sum(cross_val_score(clf.best_estimator_, X_train, y_train, cv=10, scoring="accuracy")) / 10
    dic['accuracy_val'] = sum(cross_val_score(clf.best_estimator_, X_val, y_val, cv=10, scoring="accuracy")) / 10
    y_probas_val = cross_val_predict(clf.best_estimator_, X_val, y_val, cv=5, method="predict_proba")
    y_scores_val = y_probas_val[:, 1]  # score = proba of positive class
    y_probas_train = cross_val_predict(clf.best_estimator_, X_train, y_train, cv=5, method="predict_proba")
    y_scores_train = y_probas_train[:, 1]
    dic['AUC_train'] = roc_auc_score(y_train, y_scores_train)
    dic['AUC_val'] = roc_auc_score(y_val, y_scores_val)
    fpr, tpr, thresholds = roc_curve(y_val, y_scores_val)
    dic['fpr'] = fpr
    dic['tpr'] = tpr
    dic_data = {}
    dic_data['y_val'] = list(y_val)
    dic_data['y_pred'] = list(clf.best_estimator_.predict(X_val))
    return dic, dic_data


def train_DecisionTreeClassifier(X_train, X_val, y_train, y_val):
    start = time.perf_counter()
    model = DecisionTreeClassifier(random_state=20)
    parameters = {
        "splitter": ("best", "random")
        # , "criterion": ("gini", "entropy")
        , "max_depth": [*range(1, 10)]
        , "min_samples_leaf": [*range(1, 50, 5)]
        , "min_impurity_decrease": [*np.linspace(0, 0.5, 20)]
    }
    clf = GridSearchCV(model, parameters, cv=10, verbose=1, n_jobs=2)
    clf.fit(X_train, y_train)
    end = time.perf_counter()
    print('DTC Running time: %s Seconds' % (end - start))
    joblib.dump(clf.best_estimator_, 'DTC_best.pkl')
    dic = {}
    dic['accuracy_train'] = sum(cross_val_score(clf.best_estimator_, X_train, y_train, cv=10, scoring="accuracy")) / 10
    dic['accuracy_val'] = sum(cross_val_score(clf.best_estimator_, X_val, y_val, cv=10, scoring="accuracy")) / 10
    y_probas_val = cross_val_predict(clf.best_estimator_, X_val, y_val, cv=5, method="predict_proba")
    y_scores_val = y_probas_val[:, 1]  # score = proba of positive class
    y_probas_train = cross_val_predict(clf.best_estimator_, X_train, y_train, cv=5, method="predict_proba")
    y_scores_train = y_probas_train[:, 1]
    dic['AUC_train'] = roc_auc_score(y_train, y_scores_train)
    dic['AUC_val'] = roc_auc_score(y_val, y_scores_val)
    fpr, tpr, thresholds = roc_curve(y_val, y_scores_val)
    dic['fpr'] = fpr
    dic['tpr'] = tpr
    dic_data = {}
    dic_data['y_val'] = list(y_val)
    dic_data['y_pred'] = list(clf.best_estimator_.predict(X_val))
    return dic, dic_data


# 20min
def train_LGBMClassifier(X_train, X_val, y_train, y_val, ):
    start = time.perf_counter()
    hyper_space = {'n_estimators': [50, 100, 200],
                   'max_depth': [4, 5, 8, -1],
                   'num_leaves': [15, 31, 63, 127],
                   'subsample': [0.6, 0.7, 0.8, 1.0],
                   'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
                   'learning_rate': [0.01, 0.02, 0.03]
                   }
    est = lgb.LGBMClassifier(n_jobs=-1, random_state=2018)
    clf = GridSearchCV(est, hyper_space, scoring='r2', cv=4, verbose=1)
    clf.fit(X_train, y_train)
    end = time.perf_counter()
    print('LGMC Running time: %s Seconds' % (end - start))
    joblib.dump(clf.best_estimator_, 'LGMC_best.pkl')
    dic = {}
    dic['accuracy_train'] = sum(cross_val_score(clf.best_estimator_, X_train, y_train, cv=10, scoring="accuracy")) / 10
    dic['accuracy_val'] = sum(cross_val_score(clf.best_estimator_, X_val, y_val, cv=10, scoring="accuracy")) / 10
    y_probas_val = cross_val_predict(clf.best_estimator_, X_val, y_val, cv=5, method="predict_proba")
    y_scores_val = y_probas_val[:, 1]  # score = proba of positive class
    y_probas_train = cross_val_predict(clf.best_estimator_, X_train, y_train, cv=5, method="predict_proba")
    y_scores_train = y_probas_train[:, 1]
    dic['AUC_train'] = roc_auc_score(y_train, y_scores_train)
    dic['AUC_val'] = roc_auc_score(y_val, y_scores_val)
    fpr, tpr, thresholds = roc_curve(y_val, y_scores_val)
    dic['fpr'] = fpr
    dic['tpr'] = tpr
    dic_data = {}
    dic_data['y_val'] = list(y_val)
    dic_data['y_pred'] = list(clf.best_estimator_.predict(X_val))
    return dic, dic_data


def train_BaggingClassifier(X_train, X_val, y_train, y_val):
    start = time.perf_counter()
    # estimators = [('rf', RandomForestRegressor(n_jobs=-1)),
    #               ('svr', SVR(kernel="rbf")),
    #               ('gdbt', GradientBoostingRegressor()),
    #               ('knn', KNeighborsRegressor(n_jobs=-1))]
    clf = BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=40)
    clf.fit(X_train, y_train)
    end = time.perf_counter()
    print('bagging Running time: %s Seconds' % (end - start))
    joblib.dump(clf, 'bagging_best.pkl')
    dic = {}
    dic['accuracy_train'] = sum(cross_val_score(clf, X_train, y_train, cv=10, scoring="accuracy")) / 10
    dic['accuracy_val'] = sum(cross_val_score(clf, X_val, y_val, cv=10, scoring="accuracy")) / 10
    y_scores_val = cross_val_predict(clf, X_val, y_val, cv=5, method="decision_function")
    y_scores_train = cross_val_predict(clf, X_train, y_train, cv=5, method="decision_function")
    dic['AUC_train'] = roc_auc_score(y_train, y_scores_train)
    dic['AUC_val'] = roc_auc_score(y_val, y_scores_val)
    fpr, tpr, thresholds = roc_curve(y_val, y_scores_val)
    dic['fpr'] = fpr
    dic['tpr'] = tpr
    dic_data = {}
    dic_data['y_val'] = list(y_val)
    dic_data['y_pred']=list(clf.predict(X_val))
    return dic, dic_data


# 加载数据并标准化
def load_data(X, y):
    scaler_train = StandardScaler()
    scaler_train.fit(X)
    X = scaler_train.transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=49)
    return X_train, X_val, y_train, y_val

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')  # dashed diagonal
    plt.axis([0, 1, 0, 1])  # Not shown in the book
    plt.xlabel('False Positive Rate', fontsize=16)  # Not shown
    plt.ylabel('True Positive Rate', fontsize=16)  # Not shown
    plt.grid(True)

def plot_classifier_roc(fpr, tpr):
    # plot
    plt.figure(figsize=(8, 6))  # Not shown
    plot_roc_curve(fpr, tpr, label='sgd')
    plt.legend(loc='lower right')
    plt.title("Roc Curve", fontsize=14)
    plt.show()




#需要改
# labels = ['0','1']
# tick_marks = np.array(range(len(labels))) + 0.5
# def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Reds):
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title,fontsize=28)
#     plt.colorbar()
#     xlocations = np.array(range(len(labels)))
#     plt.xticks(xlocations, labels)
#     plt.yticks(xlocations, labels)
#     plt.ylabel('True label',fontsize=28)
#     plt.xlabel('Predicted label',fontsize=28)
# # 画混淆矩阵
def plot_matrix(dic_data,title,labels):

    cm = confusion_matrix(dic_data['y_val'], dic_data['y_pred'])
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_normalized)
    plt.figure(figsize=(12, 8), dpi=120)

    ind_array = np.arange(len(labels))
    m, n = np.meshgrid(ind_array, ind_array)

    for m_val, n_val in zip(m.flatten(), n.flatten()):
        c = cm_normalized[m_val][n_val]
        if c > 0.01:
            plt.text(m_val, n_val, "%0.2f" % (c,), color='black', fontsize=20, va='center', ha='center')
    # offset the tick
    tick_marks = np.array(range(len(labels))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    # plot_confusion_matrix(cm_normalized, title=title)
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title(title, fontsize=28)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label', fontsize=28)
    plt.xlabel('Predicted label', fontsize=28)

    # show confusion matrix
    plt.savefig('confusion_matrix.png', format='png')
    plt.show()

dict3, dic_data = train_KNeighborsClassifier(X_train, X_val, y_train, y_val)
# labels=['0','1']
# plot_matrix(dic_data,title='XGB',labels=['0','1'])



# # 导入数据
import 问题一和问题二的数据 as sj

df = sj.load_data3()
X = df.iloc[:, 0:30]
y = df.iloc[:, -1]
# 变量标准
scaler_train = StandardScaler()
scaler_train.fit(X)
all_data = scaler_train.transform(X)
X_train, X_val, y_train, y_val = train_test_split(all_data, y, test_size=0.2, random_state=49)

