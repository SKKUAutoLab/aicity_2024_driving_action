import os
import shutil

if not os.path.exists('checkpoint_x3d'):
    os.makedirs('checkpoint_submit')
list_view = ['dash', 'rear', 'right']
list_fold = ['0', '1', '2', '3', '4']
for view in range(len(list_view)):
    for fold in range(len(list_fold)):
        name_file = os.rename(list_view[view] + '_' + list_fold[fold] + '/checkpoints/checkpoint_epoch_00030.pyth', list_view[view] + '_' + list_fold[fold] + '/checkpoints/' + list_view[view] + '_' + list_fold[fold] + '.pyth')
        shutil.copy(list_view[view] + '_' + list_fold[fold] + '/checkpoints/' + list_view[view] + '_' + list_fold[fold] + '.pyth', 'checkpoint_x3d')
