from mspca import mspca
import os
import scipy.io
import numpy as np
folder_path = 'D:/BaiduNetdiskDownload/SEED_IV/SEED_IV/eegdata/'
file_name = '1_1.mat'
people_name = ['1_1', '1_2', '1_3',
               '2_1', '2_2', '2_3',
               '3_1', '3_2', '3_3',
               '4_1', '4_2', '4_3',
               '5_1', '5_2', '5_3',
               '6_1', '6_2', '6_3',
               '7_1', '7_2', '7_3',
               '8_1', '8_2', '8_3',
               '9_1', '9_2', '9_3',
               '10_1', '10_2', '10_3',
               '11_1', '11_2', '11_3',
               '12_1', '12_2', '12_3',
               '13_1', '13_2', '13_3',
               '14_1', '14_2', '14_3',
               '15_1', '15_2', '15_3']

short_name = ['cz', 'cz', 'cz', 'ha', 'ha', 'ha', 'hql', 'hql', 'hql',
              'ldy', 'ldy', 'ldy', 'ly', 'ly', 'ly', 'mhw', 'mhw', 'mhw',
              'mz', 'mz', 'mz', 'qyt', 'qyt', 'qyt', 'rx', 'rx', 'rx',
              'tyc', 'tyc', 'tyc', 'whh', 'whh', 'whh', 'wll', 'wll', 'wll',
              'wq', 'wq', 'wq', 'zjd', 'zjd', 'zjd','zjy','zjy','zjy']
processed_data = {}
for i in range(len(people_name)):
    file_path = os.path.join(folder_path, people_name[i] + '.mat')
    data = scipy.io.loadmat(file_path)
    for j in range(24):
        array = data[short_name[i] + '_eeg' + str(j + 1)]
        mymodel = mspca.MultiscalePCA()
        print('processing:111')
        X_pred = mymodel.fit_transform(array, wavelet_func='db4', threshold=0.3)
        processed_data[short_name[i] + '_eeg' + str(j + 1)]=X_pred
    save_path= os.path.join(folder_path+ people_name[i] + 'MSPCA.npy')
    np.save(save_path, processed_data)
