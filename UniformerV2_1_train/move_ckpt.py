import os
import shutil

if not os.path.exists('checkpoint_uniformerv2_full'):
    os.makedirs('checkpoint_uniformerv2_full')
list_view = ['dash', 'rear', 'right']
list_fold = ['0', '1', '2', '3', '4']
for view in range(len(list_view)):
    for fold in range(len(list_fold)):
        os.rename(list_view[view] + '_' + list_fold[fold] + '/best.pyth', list_view[view] + '_' + list_fold[fold] + '/' + list_view[view] + '_' + list_fold[fold] + '.pyth')
        shutil.copy(list_view[view] + '_' + list_fold[fold] + '/' + list_view[view] + '_' + list_fold[fold] + '.pyth', 'checkpoint_uniformerv2_full/' + list_view[view] + '_' + list_fold[fold] + '.pyth')
