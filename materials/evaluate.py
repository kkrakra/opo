# -*- coding: utf-8 -*-
import argparse
import json
import numpy as np


def f1_score(y_true, y_pred) -> np.float32:
        
    
    tp = np.sum(y_true & y_pred)
    tn = np.sum(~y_true & ~y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* precision*recall / ( precision + recall + epsilon)

    return f1 

# Cкрипт считает метрику F1-score по которой оценивается решение
def evaluate(true, predictions):
    # Аргументы - файлы .npz
    # В true записаны правильные ответы. В pred - соответствующие предсказания
        
    assert set(true.files) == set(predictions.files), 'Not all images have predictions!'
    
    
    
    f1_scores = []
    for key in true.files:
        pred_sample = predictions[key].reshape(-1)
        true_sample = true[key].reshape(-1)

        f1_img = f1_score(true_sample,pred_sample)
        f1_scores.append(f1_img)
    
        
    return round(np.mean(f1_scores),5)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_path', type=str, default='true.npz', help="the path to the file with the gold answers")
    parser.add_argument('--pred_path', type=str, default='prediction.npz', help="the path to the file with the predictions")

    args = parser.parse_args()
    
    loaded_true = np.load(args.ref_path)
    loaded_pred = np.load(args.pred_path)
    
    
    f1_value = evaluate(loaded_true, loaded_pred)

    print(f1_value)
