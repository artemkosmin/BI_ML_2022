import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score


    tr_pos = (y_pred[y_pred == '1'] == y_true[y_pred == '1']).sum()
    tr_neg = (y_pred[y_pred == '0'] == y_true[y_pred == '0']).sum()
    
    fal_pos = (y_pred[y_pred == '1'] != y_true[y_pred == '1']).sum()
    fal_neg = (y_pred[y_pred == '0'] != y_true[y_pred == '0']).sum()
    
    precision = tr_pos/(tr_pos + fal_pos) if tr_pos != 0 else 0
    
    recall = tr_pos/(tr_pos + fal_neg) if tr_pos != 0 else 0
    
    f1 = 2 * (precision * recall)/(precision + recall) if (precision + recall) != 0 else 0
    
    accuracy = (tr_pos + tr_neg) / (tr_pos + tr_neg + fal_pos + fal_neg) if (tr_pos + tr_neg) != 0 else 0
    
    return precision, recall, f1, accuracy



def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    tp_tn_fp_fn = np.unique(y_pred == y_true, return_counts=True)[1][1] + np.unique(y_pred == y_true, return_counts=True)[1][0]
    tp_tn = np.unique(y_pred == y_true, return_counts=True)[1][1]
    
    accuracy = tp_tn / tp_tn_fp_fn
    
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    rss = np.sum(np.power(y_true - y_pred, 2))
    tss = np.sum(np.power(y_true - np.mean(y_true), 2))
    
    r_squared = 1 - (rss/tss)
    
    return r_squared

def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = 1/len(y_pred) * np.sum((np.power(y_true - y_pred, 2)))
    
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mae = 1/len(y_pred) * np.sum(np.abs(y_true - y_pred))
    
    return mae
