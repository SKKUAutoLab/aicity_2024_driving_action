import os
import traceback
import random
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")

def get_args():
    parser = ArgumentParser('Track 3 evaluation')
    parser.add_argument("--predictions_file", type=str, help="path to predictions file", default='pred.txt')
    parser.add_argument("--ground_truth_file", type=str, help="path to ground truth file", default='GT.txt')
    return parser.parse_args()

def getData(fpath, names=None, sep='\s+|\t+|,'):
    try:
        df = pd.read_csv(fpath, sep=sep, index_col=None, skipinitialspace=True, header=None, names=names, engine='python')
        return df.sort_values(by=df.columns[0:3].tolist(), ascending=[True, True, True])
    except Exception as e:
        raise ValueError("Could not read input from %s. Error: %s" % (os.path.basename(fpath), repr(e)))

def print_results(df):
    print(df.to_string(index=False, float_format='{:2.4f}'.format, col_space=7, justify='left'))

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
        return max(min(ge, pe) - max(gs, ps), 0) / (max(ge, pe)-min(gs, ps)) # os(p, g)

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
    return pd.DataFrame([(mos,)], columns=['mOS']), mos

# if __name__ == '__main__':
#     args = get_args()
#     random.seed(10)
#     labels = getData(args.ground_truth_file, names=['video_id', 'activity_id', 'ts_start', 'ts_end'])
#     labels = labels[labels.activity_id != 0]
#     try:
#         predictions = getData(args.predictions_file, names=['video_id', 'activity_id', 'ts_start', 'ts_end'])
#         predictions = predictions[predictions.activity_id != 0]
#         with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
#             print(predictions)
#         df = eval_ndar(labels, predictions)
#         print_results(df)
#     except Exception as e:
#         print("Error: %s" % repr(e))
#         traceback.print_exc()

if __name__ == '__main__':
    args = get_args()
    random.seed(10)
    labels = getData(args.ground_truth_file, names=['video_id', 'activity_id', 'ts_start', 'ts_end'])
    labels = labels[labels.activity_id != 0]
    max_mos = 0
    best_file_refine = ''
    try:
        import glob
        from natsort import natsorted
        # list_output_refine = glob.glob('list_output_refine/*.txt')
        list_output_refine = os.listdir('list_output_refine')
        for i in range(len(list_output_refine)):
            predictions = getData('list_output_refine/' + list_output_refine[i], names=['video_id', 'activity_id', 'ts_start', 'ts_end'])
            predictions = predictions[predictions.activity_id != 0]
            df, mos = eval_ndar(labels, predictions)
            if max_mos < mos:
                max_mos = mos
                best_file_refine = list_output_refine[i]
    except Exception as e:
        print("Error: %s" % repr(e))
        traceback.print_exc()
    print(best_file_refine)
    print(max_mos)