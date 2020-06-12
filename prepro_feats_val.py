"""
Preprocess a raw json dataset into features files for use in data_loader.py

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: two folders of features
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
from six.moves import cPickle
import numpy as np
import torch
import torchvision.models as models
import skimage.io

from torchvision import transforms as trn
preprocess = trn.Compose([
                #trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from misc.resnet_utils import myResnet
import misc.resnet as resnet

from scipy.io import loadmat
import hdf5storage

def main(params):
    net = getattr(resnet, 'resnet152')()
    num_ftrs = net.fc.in_features
#    net.fc = torch.nn.Linear(num_ftrs, 1)
    net.load_state_dict(torch.load('./resnet152.pth'),False)
    my_resnet = myResnet(net)
    my_resnet.cuda()
    my_resnet.eval()
    if not os.path.exists('resnet152'):
        subprocess.call('mkdir resnet152', shell=True)


    train_set = hdf5storage.read(path='/img_path',
                         filename='./val_set.h5')
    print(len(train_set))
    seed(123)  # make reproducible
    for i in range(8000+780,len(train_set)):
        outputs = []
        
        for j in range(8):
            I = skimage.io.imread('./' + train_set[i][j].replace('\\', '/'),as_gray=1)
            if len(I.shape) == 2:
                I = I[:, :, np.newaxis]
                I = np.concatenate((I, I, I), axis=2)
            I = I.astype('float32') / 255.0
            I = torch.from_numpy(I.transpose([2, 0, 1])).cuda()
            I = preprocess(I)
            with torch.no_grad():
                tmp_fc, tmp_att = my_resnet(I, params['att_size'])
            outputs.append(tmp_fc.data.cpu().float().numpy())
        file_nm=os.path.join('./resnet152','v_video'+str(i+1)+'.npy')
        np.save(file_nm,outputs)
        print(file_nm)

  


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', required=0, help='input json file to process into hdf5')
    parser.add_argument('--output_dir', default='data', help='output h5 file')

    # options
    parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--att_size', default=14, type=int, help='14x14 or 7x7')
    parser.add_argument('--model', default='resnet101', type=str, help='resnet101, resnet152')
    parser.add_argument('--model_root', default='./data/imagenet_weights', type=str, help='model root')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main(params)
