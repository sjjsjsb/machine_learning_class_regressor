#回归算法lgb
import time

import numpy as np
import pandas as pd
import xgboost as xgb


from matplotlib import pyplot as plt
from scipy.stats import reciprocal, uniform, randint
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, BaggingRegressor, \
    HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV, cross_val_score
import multiprocessing
import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import lightgbm as lgb
from deepforest import CascadeForestRegressor
import optuna
from optuna.samplers import TPESampler




def train_xgb_regressor(X_train, X_val, y_train, y_val):
    start = time.perf_counter()
    xgb_model = xgb.XGBRegressor(n_jobs=multiprocessing.cpu_count() // 2)
    clf = GridSearchCV(xgb_model, {'max_depth': [2, 4, 6],
                                   'n_estimators': [50, 100, 200]}, verbose=1, n_jobs=2)
    clf.fit(X_train, y_train)
    joblib.dump(clf.best_estimator_, 'xgbR_best.pkl')
    end = time.perf_counter()
    print('xgbR Running time: %s Seconds' % (end - start))
    y_train_pred = clf.best_estimator_.predict(X_train)
    y_val_pred = clf.best_estimator_.predict(X_val)
    get_error = {}
    # train_MSE =mean_absolute_error(y_train,y_train_pred)
    # train_RMSE=np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    # defen=np.sqrt(clf.score(X_val,y_val))
    # val_MSE = mean_absolute_error(y_val, y_val_pred)
    # val_R2 = r2_score(y_val, y_val_pred)
    # val_RMSE=np.sqrt(np.mean((y_val - y_val_pred) ** 2))
    get_error['train_MSE'] = mean_absolute_error(y_train, y_train_pred)
    get_error['train_RMSE'] = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    get_error['defen'] = np.sqrt(clf.score(X_val, y_val))
    get_error['val_MSE'] = mean_absolute_error(y_val, y_val_pred)
    get_error['val_R2'] = r2_score(y_val, y_val_pred)
    get_error['val_RMSE'] = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
    # 显示中文
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 画出图形
    aa_true = list(y_val[:30])
    aa_val = list(y_val_pred[:30])
    plt.figure(figsize=(10, 5.5))
    plt.plot(aa_true, "b.-", label="ture")
    plt.plot(aa_val, "r.-", label="val")
    plt.xlabel("30 个数据编号")
    # 可能要改的值#######################################
    plt.ylabel("pIC50")
    plt.legend(['test_true', 'test_pre'])
    plt.title("xgbR 30 个测试数据真实值和预测值", fontsize=14)
    plt.savefig('xgbR 30 个测试数据真实值和预测值.png', dpi=500, bbox_inches='tight')  # 解决图 片不清晰，不完整的问题
    plt.show()

    return get_error


def train_svr(X_train, X_val, y_train, y_val):
    start = time.perf_counter()
    svm_clf = SVR(kernel='rbf')
    param_distributions = {"gamma": reciprocal(0.001, 0.01), "C": uniform(1, 10)}
    clf = RandomizedSearchCV(svm_clf, param_distributions, n_iter=100, verbose=2, cv=5, scoring='r2',
                             random_state=42, n_jobs=1)
    clf.fit(X_train, y_train)
    end = time.perf_counter()
    print('svr Running time: %s Seconds' % (end - start))
    joblib.dump(clf.best_estimator_, 'svr_best.pkl')
    y_train_pred = clf.best_estimator_.predict(X_train)
    y_val_pred = clf.best_estimator_.predict(X_val)
    get_error = {}
    get_error['train_MSE'] = mean_absolute_error(y_train, y_train_pred)
    get_error['train_RMSE'] = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    get_error['defen'] = np.sqrt(clf.score(X_val, y_val))
    get_error['val_MSE'] = mean_absolute_error(y_val, y_val_pred)
    get_error['val_R2'] = r2_score(y_val, y_val_pred)
    get_error['val_RMSE'] = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
    # 显示中文
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 画出图形
    aa_true = list(y_val[:30])
    aa_val = list(y_val_pred[:30])
    plt.figure(figsize=(10, 5.5))
    plt.plot(aa_true, "b.-", label="ture")
    plt.plot(aa_val, "r.-", label="val")
    plt.xlabel("30 个数据编号")
    # 可能要改的值#######################################
    plt.ylabel("pIC50")
    plt.legend(['test_true', 'test_pre'])
    plt.title("svr 30 个测试数据真实值和预测值", fontsize=14)
    plt.savefig('svr 30 个测试数据真实值和预测值.png', dpi=500, bbox_inches='tight')  # 解决图 片不清晰，不完整的问题
    plt.show()
    return get_error


def train_rf_regressor(X_train, X_val, y_train, y_val):
    start = time.perf_counter()
    forest_reg = RandomForestRegressor(random_state=42)
    param_distribs = {'n_estimators': randint(low=1, high=200), 'max_features': randint(low=1, high=8), }
    clf = RandomizedSearchCV(forest_reg, param_distributions=param_distribs, n_iter=100, cv=5, scoring='r2',
                             random_state=42)
    clf.fit(X_train, y_train)
    end = time.perf_counter()
    print('rfR Running time: %s Seconds' % (end - start))
    joblib.dump(clf.best_estimator_, 'rfR_best.pkl')
    y_train_pred = clf.best_estimator_.predict(X_train)
    y_val_pred = clf.best_estimator_.predict(X_val)
    get_error = {}
    get_error['train_MSE'] = mean_absolute_error(y_train, y_train_pred)
    get_error['train_RMSE'] = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    get_error['defen'] = np.sqrt(clf.score(X_val, y_val))
    get_error['val_MSE'] = mean_absolute_error(y_val, y_val_pred)
    get_error['val_R2'] = r2_score(y_val, y_val_pred)
    get_error['val_RMSE'] = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
    # 显示中文
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 画出图形
    aa_true = list(y_val[:30])
    aa_val = list(y_val_pred[:30])
    plt.figure(figsize=(10, 5.5))
    plt.plot(aa_true, "b.-", label="ture")
    plt.plot(aa_val, "r.-", label="val")
    plt.xlabel("30 个数据编号")
    # 可能要改的值#######################################
    plt.ylabel("pIC50")
    plt.legend(['test_true', 'test_pre'])
    plt.title("rfR 30 个测试数据真实值和预测值", fontsize=14)
    plt.savefig('rfR 30 个测试数据真实值和预测值.png', dpi=500, bbox_inches='tight')  # 解决图 片不清晰，不完整的问题
    plt.show()
    return get_error


def train_GBRT_regressor(X_train, X_val, y_train, y_val):
    start = time.perf_counter()
    gbrt = GradientBoostingRegressor(random_state=42, learning_rate=0.1)
    param_distribs = {'max_depth': randint(low=2, high=20), 'n_estimators': randint(low=20, high=200), }
    clf = RandomizedSearchCV(gbrt, param_distributions=param_distribs, n_iter=100, cv=5, scoring='r2',
                             random_state=42)
    clf.fit(X_train, y_train)
    end = time.perf_counter()
    print('gbrtR Running time: %s Seconds' % (end - start))
    joblib.dump(clf.best_estimator_, 'gbrtR_best.pkl')
    y_train_pred = clf.best_estimator_.predict(X_train)
    y_val_pred = clf.best_estimator_.predict(X_val)
    get_error = {}
    get_error['train_MSE'] = mean_absolute_error(y_train, y_train_pred)
    get_error['train_RMSE'] = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    get_error['defen'] = np.sqrt(clf.score(X_val, y_val))
    get_error['val_MSE'] = mean_absolute_error(y_val, y_val_pred)
    get_error['val_R2'] = r2_score(y_val, y_val_pred)
    get_error['val_RMSE'] = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
    # 显示中文
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 画出图形
    aa_true = list(y_val[:30])
    aa_val = list(y_val_pred[:30])
    plt.figure(figsize=(10, 5.5))
    plt.plot(aa_true, "b.-", label="ture")
    plt.plot(aa_val, "r.-", label="val")
    plt.xlabel("30 个数据编号")
    # 可能要改的值#######################################
    plt.ylabel("pIC50")
    plt.legend(['test_true', 'test_pre'])
    plt.title("gbrtR 30 个测试数据真实值和预测值", fontsize=14)
    plt.savefig('gbrtR 30 个测试数据真实值和预测值.png', dpi=500, bbox_inches='tight')  # 解决图 片不清晰，不完整的问题
    plt.show()
    return get_error


def train_KNeighborsRegressor(X_train, X_val, y_train, y_val):
    start = time.perf_counter()
    model = KNeighborsRegressor(n_jobs=-1)
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
    print('knnR Running time: %s Seconds' % (end - start))
    joblib.dump(clf.best_estimator_, 'knnR_best.pkl')
    y_train_pred = clf.best_estimator_.predict(X_train)
    y_val_pred = clf.best_estimator_.predict(X_val)
    get_error = {}
    get_error['train_MSE'] = mean_absolute_error(y_train, y_train_pred)
    get_error['train_RMSE'] = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    get_error['defen'] = np.sqrt(clf.score(X_val, y_val))
    get_error['val_MSE'] = mean_absolute_error(y_val, y_val_pred)
    get_error['val_R2'] = r2_score(y_val, y_val_pred)
    get_error['val_RMSE'] = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
    # 显示中文
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 画出图形
    aa_true = list(y_val[:30])
    aa_val = list(y_val_pred[:30])
    plt.figure(figsize=(10, 5.5))
    plt.plot(aa_true, "b.-", label="ture")
    plt.plot(aa_val, "r.-", label="val")
    plt.xlabel("30 个数据编号")
    # 可能要改的值#######################################
    plt.ylabel("pIC50")
    plt.legend(['test_true', 'test_pre'])
    plt.title("knnR 30 个测试数据真实值和预测值", fontsize=14)
    plt.savefig('knnR 30 个测试数据真实值和预测值.png', dpi=500, bbox_inches='tight')  # 解决图 片不清晰，不完整的问题
    plt.show()
    return get_error



def train_StackingRegressor(X_train,X_val,y_train,y_val):
    start = time.perf_counter()
    estimators = [('rf', RandomForestRegressor(n_jobs=-1)),
                  ('svr', SVR(kernel="rbf")),
                  ('gdbt', GradientBoostingRegressor()),
                  ('knn', KNeighborsRegressor(n_jobs=-1))]
    clf=StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
    clf.fit(X_train, y_train)
    end = time.perf_counter()
    print('StackR Running time: %s Seconds' % (end - start))
    joblib.dump(clf, 'StackR_best.pkl')
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    get_error = {}
    get_error['train_MSE'] = mean_absolute_error(y_train, y_train_pred)
    get_error['train_RMSE'] = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    get_error['defen'] = np.sqrt(clf.score(X_val, y_val))
    get_error['val_MSE'] = mean_absolute_error(y_val, y_val_pred)
    get_error['val_R2'] = r2_score(y_val, y_val_pred)
    get_error['val_RMSE'] = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
    # 显示中文
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 画出图形
    aa_true = list(y_val[:30])
    aa_val = list(y_val_pred[:30])
    plt.figure(figsize=(10, 5.5))
    plt.plot(aa_true, "b.-", label="ture")
    plt.plot(aa_val, "r.-", label="val")
    plt.xlabel("30 个数据编号")
    # 可能要改的值#######################################
    plt.ylabel("pIC50")
    plt.legend(['test_true', 'test_pre'])
    plt.title("StackR 30 个测试数据真实值和预测值", fontsize=14)
    plt.savefig('StackR 30 个测试数据真实值和预测值.png', dpi=500, bbox_inches='tight')  # 解决图 片不清晰，不完整的问题
    plt.show()
    return get_error


def train_MLP(X_train, X_val, y_train, y_val):
    start = time.perf_counter()
    parameters = {'hidden_layer_sizes': [200, 250, 300, 400, 500, 600], 'activation': ['relu']}
    model= MLPRegressor(
     solver='adam',
        # '''，激活函数用relu，梯度下降方法用adam'''
    alpha = 0.01, max_iter = 500)
    '''惩罚系数为0.01，最大迭代次数为200'''
    clf = GridSearchCV(model, parameters, cv=10, n_jobs=4, scoring='r2')
    clf.fit(X_train, y_train)
    end = time.perf_counter()
    print('MLP Running time: %s Seconds' % (end - start))
    joblib.dump(clf.best_estimator_, 'MLP_best.pkl')
    y_train_pred = clf.best_estimator_.predict(X_train)
    y_val_pred = clf.best_estimator_.predict(X_val)
    get_error = {}
    get_error['train_MSE'] = mean_absolute_error(y_train, y_train_pred)
    get_error['train_RMSE'] = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    get_error['defen'] = np.sqrt(clf.score(X_val, y_val))
    get_error['val_MSE'] = mean_absolute_error(y_val, y_val_pred)
    get_error['val_R2'] = r2_score(y_val, y_val_pred)
    get_error['val_RMSE'] = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
    # 显示中文
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 画出图形
    aa_true = list(y_val[:30])
    aa_val = list(y_val_pred[:30])
    plt.figure(figsize=(10, 5.5))
    plt.plot(aa_true, "b.-", label="ture")
    plt.plot(aa_val, "r.-", label="val")
    plt.xlabel("30 个数据编号")
    # 可能要改的值#######################################
    plt.ylabel("pIC50")
    plt.legend(['test_true', 'test_pre'])
    plt.title("MLP 30 个测试数据真实值和预测值", fontsize=14)
    plt.savefig('MLP 30 个测试数据真实值和预测值.png', dpi=500, bbox_inches='tight')  # 解决图 片不清晰，不完整的问题
    plt.show()
    return get_error


def train_LinerRegression(X_train, X_val, y_train, y_val):
    start = time.perf_counter()
    clf =LinearRegression()
    clf.fit(X_train, y_train)
    end = time.perf_counter()
    print('LRR Running time: %s Seconds' % (end - start))
    joblib.dump(clf, 'LRR_best.pkl')
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    get_error = {}
    get_error['train_MSE'] = mean_absolute_error(y_train, y_train_pred)
    get_error['train_RMSE'] = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    get_error['defen'] = np.sqrt(clf.score(X_val, y_val))
    get_error['val_MSE'] = mean_absolute_error(y_val, y_val_pred)
    get_error['val_R2'] = r2_score(y_val, y_val_pred)
    get_error['val_RMSE'] = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
    # 显示中文
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 画出图形
    aa_true = list(y_val[:30])
    aa_val = list(y_val_pred[:30])
    plt.figure(figsize=(10, 5.5))
    plt.plot(aa_true, "b.-", label="ture")
    plt.plot(aa_val, "r.-", label="val")
    plt.xlabel("30 个数据编号")
    # 可能要改的值#######################################
    plt.ylabel("pIC50")
    plt.legend(['test_true', 'test_pre'])
    plt.title("LRR 30 个测试数据真实值和预测值", fontsize=14)
    plt.savefig('LRR 30 个测试数据真实值和预测值.png', dpi=500, bbox_inches='tight')  # 解决图 片不清晰，不完整的问题
    plt.show()
    return get_error


def train_DecisionTreeRegressor(X_train, X_val, y_train, y_val):
    start = time.perf_counter()
    model = DecisionTreeRegressor(random_state=20)
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
    print('DTR Running time: %s Seconds' % (end - start))
    joblib.dump(clf.best_estimator_, 'DTR_best.pkl')
    y_train_pred = clf.best_estimator_.predict(X_train)
    y_val_pred = clf.best_estimator_.predict(X_val)
    get_error = {}
    get_error['train_MSE'] = mean_absolute_error(y_train, y_train_pred)
    get_error['train_RMSE'] = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    get_error['defen'] = np.sqrt(clf.score(X_val, y_val))
    get_error['val_MSE'] = mean_absolute_error(y_val, y_val_pred)
    get_error['val_R2'] = r2_score(y_val, y_val_pred)
    get_error['val_RMSE'] = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
    # 显示中文
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 画出图形
    aa_true = list(y_val[:30])
    aa_val = list(y_val_pred[:30])
    plt.figure(figsize=(10, 5.5))
    plt.plot(aa_true, "b.-", label="ture")
    plt.plot(aa_val, "r.-", label="val")
    plt.xlabel("30 个数据编号")
    # 可能要改的值#######################################
    plt.ylabel("pIC50")
    plt.legend(['test_true', 'test_pre'])
    plt.title("DTR 30 个测试数据真实值和预测值", fontsize=14)
    plt.savefig('DTR 30 个测试数据真实值和预测值.png', dpi=500, bbox_inches='tight')  # 解决图 片不清晰，不完整的问题
    plt.show()
    return get_error

def train_LGBMRegressor(X_train, X_val, y_train, y_val):
    start = time.perf_counter()
    hyper_space = {'n_estimators': [50, 100, 200],
                   'max_depth': [4, 5, 8, -1],
                   'num_leaves': [15, 31, 63, 127],
                   'subsample': [0.6, 0.7, 0.8, 1.0],
                   'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
                   'learning_rate': [0.01, 0.02, 0.03]
                   }
    est = lgb.LGBMRegressor(n_jobs=-1, random_state=2018)
    clf= GridSearchCV(est, hyper_space, scoring='r2', cv=4, verbose=1)
    clf.fit(X_train, y_train)
    end = time.perf_counter()
    print('LGMR Running time: %s Seconds' % (end - start))
    joblib.dump(clf.best_estimator_, 'LGMR_best.pkl')
    y_train_pred = clf.best_estimator_.predict(X_train)
    y_val_pred = clf.best_estimator_.predict(X_val)
    get_error = {}
    get_error['train_MSE'] = mean_absolute_error(y_train, y_train_pred)
    get_error['train_RMSE'] = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    get_error['defen'] = np.sqrt(clf.score(X_val, y_val))
    get_error['val_MSE'] = mean_absolute_error(y_val, y_val_pred)
    get_error['val_R2'] = r2_score(y_val, y_val_pred)
    get_error['val_RMSE'] = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
    # 显示中文
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 画出图形
    aa_true = list(y_val[:30])
    aa_val = list(y_val_pred[:30])
    plt.figure(figsize=(10, 5.5))
    plt.plot(aa_true, "b.-", label="ture")
    plt.plot(aa_val, "r.-", label="val")
    plt.xlabel("30 个数据编号")
    # 可能要改的值#######################################
    plt.ylabel("pIC50")
    plt.legend(['test_true', 'test_pre'])
    plt.title("LGMR 30 个测试数据真实值和预测值", fontsize=14)
    plt.savefig('LGMR 30 个测试数据真实值和预测值.png', dpi=500, bbox_inches='tight')  # 解决图 片不清晰，不完整的问题
    plt.show()
    return get_error

def train_BaggingRegressor(X_train, X_val, y_train, y_val):
    start = time.perf_counter()
    # estimators = [('rf', RandomForestRegressor(n_jobs=-1)),
    #               ('svr', SVR(kernel="rbf")),
    #               ('gdbt', GradientBoostingRegressor()),
    #               ('knn', KNeighborsRegressor(n_jobs=-1))]
    clf=BaggingRegressor(base_estimator=SVR(),n_estimators=10,random_state=40)
    clf.fit(X_train, y_train)
    end = time.perf_counter()
    print('bagging Running time: %s Seconds' % (end - start))
    joblib.dump(clf, 'bagging_best.pkl')
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    get_error = {}
    get_error['train_MSE'] = mean_absolute_error(y_train, y_train_pred)
    get_error['train_RMSE'] = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    get_error['defen'] = np.sqrt(clf.score(X_val, y_val))
    get_error['val_MSE'] = mean_absolute_error(y_val, y_val_pred)
    get_error['val_R2'] = r2_score(y_val, y_val_pred)
    get_error['val_RMSE'] = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
    # 显示中文
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 画出图形
    aa_true = list(y_val[:30])
    aa_val = list(y_val_pred[:30])
    plt.figure(figsize=(10, 5.5))
    plt.plot(aa_true, "b.-", label="ture")
    plt.plot(aa_val, "r.-", label="val")
    plt.xlabel("30 个数据编号")
    # 可能要改的值#######################################
    plt.ylabel("pIC50")
    plt.legend(['test_true', 'test_pre'])
    plt.title("bagging 30 个测试数据真实值和预测值", fontsize=14)
    plt.savefig('bagging 30 个测试数据真实值和预测值.png', dpi=500, bbox_inches='tight')  # 解决图 片不清晰，不完整的问题
    plt.show()
    return get_error

# def train_HistGradientBoostingRegressor(X_train, X_val, y_train, y_val):
#     HistGradientBoostingRegressor

def train_deepforest():
    start = time.perf_counter()
    hyper_space = {'n_estimators': [2, 4, 8],
                   'n_trees':[50,100,150],
                   'max_layers':[2,4,6,8],
                   'min_samples_leaf':[1, 20, 5]
                   }
    est =CascadeForestRegressor(random_state=42)
    clf= GridSearchCV(est, hyper_space, scoring='r2', cv=4, verbose=1)
    clf.fit(X_train, y_train)
    end = time.perf_counter()
    print('LGMR Running time: %s Seconds' % (end - start))
    joblib.dump(clf.best_estimator_, 'LGMR_best.pkl')
    y_train_pred = clf.best_estimator_.predict(X_train)
    y_val_pred = clf.best_estimator_.predict(X_val)
    get_error = {}
    get_error['train_MSE'] = mean_absolute_error(y_train, y_train_pred)
    get_error['train_RMSE'] = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
    get_error['defen'] = np.sqrt(clf.score(X_val, y_val))
    get_error['val_MSE'] = mean_absolute_error(y_val, y_val_pred)
    get_error['val_R2'] = r2_score(y_val, y_val_pred)
    get_error['val_RMSE'] = np.sqrt(np.mean((y_val - y_val_pred) ** 2))
    # 显示中文
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 画出图形
    aa_true = list(y_val[:30])
    aa_val = list(y_val_pred[:30])
    plt.figure(figsize=(10, 5.5))
    plt.plot(aa_true, "b.-", label="ture")
    plt.plot(aa_val, "r.-", label="val")
    plt.xlabel("30 个数据编号")
    # 可能要改的值#######################################
    plt.ylabel("pIC50")
    plt.legend(['test_true', 'test_pre'])
    plt.title("DeeF 30 个测试数据真实值和预测值", fontsize=14)
    plt.savefig('DeeF 30 个测试数据真实值和预测值.png', dpi=500, bbox_inches='tight')  # 解决图 片不清晰，不完整的问题
    plt.show()
    return get_error



def train_all(X_train, X_val, y_train, y_val):
    dic={}
    dic['svr']= train_svr(X_train, X_val, y_train, y_val).values()
    # train_xgb_regressor
    dic['xgb']= train_xgb_regressor(X_train, X_val, y_train, y_val).values()
    # rf_regressor
    dic['rf']= train_rf_regressor(X_train, X_val, y_train, y_val)
    # GBRT
    dic['GBRT'] = train_GBRT_regressor(X_train, X_val, y_train, y_val)
    # knn_regressor
    dic['knn'] = train_KNeighborsRegressor(X_train, X_val, y_train, y_val)
    # DecisionTreeRegressor
    dic['DT']= train_DecisionTreeRegressor(X_train, X_val, y_train, y_val)
    # LGBMR
    dic['lgb']= train_LGBMRegressor(X_train, X_val, y_train, y_val)
    # LRR
    dic['lrr'] = train_LinerRegression(X_train, X_val, y_train, y_val)
    # stackR
    dic['stack']= train_StackingRegressor(X_train, X_val, y_train, y_val)
    # MLP
    dic['MLP']= train_MLP(X_train, X_val, y_train, y_val)
    # bagging
    dic['bagging'] = train_BaggingRegressor(X_train, X_val, y_train, y_val)
    # names = ['svr','xgb','rf','GBRT','KNeighborsRegressor','DT','lgm''lr','stacking','MLP','bagging']
    names=['train_MSE','train_RMSE','defen','val_MSE','val_R2','val_RMSE']
    regressor_df = pd.DataFrame(dic,index=names)
    return regressor_df