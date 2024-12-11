import numpy as np
import torch
from torcheval.metrics.functional import r2_score
from torcheval.metrics import R2Score

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true, epsilon=1e-8):
    return np.mean(np.abs((pred - true) / (true + epsilon)))


def MSPE(pred, true, epsilon=1e-8):
    return np.mean(np.square((pred - true) / (true + epsilon)))


def R2(pred, true):
    target_mean = np.mean(true)
    ss_tot = np.sum((pred - target_mean) ** 2)
    ss_res = np.sum((pred - true) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r2 = R2(pred, true)

    return mae, mse, rmse, mape, mspe, r2
