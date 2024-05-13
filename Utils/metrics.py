import numpy as np
from scipy import stats
from easydict import EasyDict as edict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred) / (y_true + 1e-5)) * 100

def return_result(y_true,y_pred):
    performance = edict()
    performance.corr = stats.pearsonr(y_true, y_pred)[0]
    performance.mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
    performance.rmse = mean_squared_error(y_true=y_true,y_pred=y_pred, squared=True)
    performance.r2 = r2_score(y_true=y_true, y_pred=y_pred)
    performance.mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    
    return performance