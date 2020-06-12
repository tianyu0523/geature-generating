import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn

from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video
import hdf5storage


if __name__=="__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 1
    opt.n_classes = 400

    model = generate_model(opt)
    model_data = torch.load('./resnext-101-64f-kinetics.pth')

    assert opt.arch == model_data['arch']

    model.load_state_dict(model_data['state_dict'],False)

    model.eval()

    class_names = []
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    if os.path.exists('tmpv11'):
        subprocess.call('rm -rf tmpv11', shell=True)
    if not os.path.exists('resnext101-64f'):
        subprocess.call('mkdir resnext101-64f', shell=True)


    val_set = hdf5storage.read(path='/img_path', filename='./val_set.h5')
    print(len(val_set))
    for i in range(len(val_set)):

        subprocess.call('mkdir tmpv11', shell=True)
        for j in range(8):
            img_path='./' + val_set[i][j].replace('\\', '/')

            subprocess.call('cp '+img_path+' tmpv11/'+'image_{:05d}.jpg'.format(j+1),shell=True)

        result = classify_video('tmpv11', str(i), class_names, model, opt)
        outputs = []
        for kk in range(7):
            outputs.append(result['clips'][kk]['features'])
        file_nm='./resnext101-64f/v_video'+str(i+1)+'.npy'
        np.save(file_nm, outputs)
        subprocess.call('rm -rf tmpv11', shell=True)
        print(file_nm)

    if os.path.exists('tmpv11'):
        subprocess.call('rm -rf tmpv11', shell=True)

    # with open(opt.output, 'w') as f:
    #     json.dump(outputs, f)
