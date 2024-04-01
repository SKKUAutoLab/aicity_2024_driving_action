import os
import traceback
import random
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

def getData(fpath, sep='\s+|\t+|,'):
    try:
        df = pd.read_csv(fpath, sep=sep, index_col=None, skipinitialspace=True, header=None, names=["video_id", "activity_id", "ts_start", "ts_end"], engine='python')
        return df.sort_values(by=['video_id', 'ts_start'])
    except Exception as e:
        raise ValueError("Could not read input from %s. Error: %s" % (os.path.basename(fpath), repr(e)))

def eval_ndar(labels, predictions):
    if predictions is None:
        return None
    if not predictions[predictions.ts_end < predictions.ts_start].empty:
        raise ValueError('End times must be greater or equal to start times.')
    predictions['found'] = False
    predictions['os'] = 0
    overlap = []
    mgt = 0

    def compute_overlap(ps, pe, gs, ge):
        return max(min(ge, pe) - max(gs, ps), 0) / (max(ge, pe) - min(gs, ps)) # os(p, g)

    for _, qrow in labels.iterrows():
        p = predictions[(predictions.found == False) & (predictions.video_id == qrow.video_id) & (predictions.activity_id == qrow.activity_id) & (predictions.ts_start >= qrow.ts_start - 10) & (predictions.ts_start <= qrow.ts_start + 10) & (predictions.ts_end >= qrow.ts_end - 10) & (predictions.ts_end <= qrow.ts_end + 10)]
        if p.empty:
            mgt += 1
            continue
        p['os'] = p.apply(lambda x: compute_overlap(x.ts_start, x.ts_end, qrow.ts_start, qrow.ts_end), axis=1)
        x = p[p.os == p.os.max()].iloc[0]
        predictions.loc[(predictions.found == False) & (predictions.video_id == x.video_id) & (predictions.activity_id == x.activity_id) & (predictions.ts_start == x.ts_start) & (predictions.ts_end == x.ts_end), 'found'] = True
        overlap.append(x.os)
    mos = np.sum(overlap) / (len(predictions) + mgt)
    # print(predictions[predictions['found']==False])

    df_true = predictions[['activity_id', 'found']].copy()
    df_pred = predictions[['activity_id', 'found']].copy()
    for i in range(len(df_pred)):
        if df_pred['found'][i] == False:
            df_pred['activity_id'][i] = 0
    y_true = df_true['activity_id'].tolist()
    y_pred = df_pred['activity_id'].tolist()
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='micro')
    matrix = confusion_matrix(y_true, y_pred)
    acc_by_class = matrix.diagonal() / matrix.sum(axis=1)
    print('Accuracy:', accuracy)
    print('F1 Score:', f1)
    print('Accuracy by class:', acc_by_class)

if __name__ == '__main__':
    random.seed(10)
    labels = getData('gt_val_fold4.txt')
    labels = labels[labels.activity_id != 0]
    predictions_file = 'pred_videomae_fold4.txt'
    print('Evaluating file:', predictions_file)
    try:
        predictions = getData(predictions_file)
        predictions = predictions[predictions.activity_id != 0]
        eval_ndar(labels, predictions)
    except Exception as e:
        print("Error: %s" % repr(e))
        traceback.print_exc()