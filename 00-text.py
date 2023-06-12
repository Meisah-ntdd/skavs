import os
import random

x = '/home/21151213641/xdhpc/voxceleb_trainer-master/datas/voxceleb1_mini'
y = 'G:/Paper/Dataset/voxeceleb/mfa_conformer-master/data/eval_wav/wav/'

dataset_path = 'G:/Paper/Dataset/voxeceleb/mfa_conformer-master/data/wav/'

voxceleb2_dev_txt = open('G:/Paper/Dataset/voxeceleb/mfa_conformer-master/data' + '/' + 'train_vox2.txt', 'w')



dataset_flag = 'dev' # dev or eval or all
line_flag = 1
key_list = []

for r, ds, fs in os.walk(dataset_path):
    for f in fs:
        fn = '/'.join([r, f]).replace('\\', '/')
        key = '/'.join(fn.split('/')[-3:])
        id = key[:7]
        if key[-3:] == 'wav': 
            voxceleb2_dev_txt.write(key + ' ' + id + '\n')


