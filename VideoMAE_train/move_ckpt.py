import os
import shutil

if not os.path.exists('checkpoint_videomae'):
    os.makedirs('checkpoint_videomae')
list_view = ['dash', 'rear', 'right']
list_fold = ['0', '1', '2', '3', '4']
for view in range(len(list_view)):
    for fold in range(len(list_fold)):
        os.rename('checkpoints/recog_' + list_view[view] + '_' + list_fold[fold] + '/checkpoint-best.pth', 'checkpoints/recog_' + list_view[view] + '_' + list_fold[fold] + '/' + list_view[view] + '_' + list_fold[fold] + '.pth')
        shutil.copy('checkpoints/recog_' + list_view[view] + '_' + list_fold[fold] + '/' + list_view[view] + '_' + list_fold[fold] + '.pth', 'checkpoint_videomae/' + list_view[view] + '_' + list_fold[fold] + '.pth')
