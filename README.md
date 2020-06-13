
## Requirements
* [PyTorch](http://pytorch.org/)
```
conda install pytorch torchvision cuda80 -c soumith scikit-image h5py  
```
* FFmpeg, FFprobe
```
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-64bit-static.tar.xz
cd ./ffmpeg-3.3.3-64bit-static/; sudo cp ffmpeg ffprobe /usr/local/bin;
```
* Python 3

## Preparation
* Download this code.
* Download pretrained model ['resnet152.pth'](https://drive.google.com/file/d/18C2YfPkj3GRWcDz7yNBAMsB1fDRCoG07/view?usp=sharing) and ['resnext-101-64f-kinetics.pth'](https://drive.google.com/file/d/1A416JN5FvDvOw8wM7D_VUZ3o4uE57iMG/view?usp=sharing). Then put them in the main folder.
* Download the [images](https://drive.google.com/file/d/1IYFQHJLuR02_5ZYS4iyZePc1EfyBNMza/view?usp=sharing), unzip and put the 'visual_data' folder in the main folder.


## Usage


Run
```
python main_img_train.py --mode feature
```
to get motion features of train set
Run
```
python main_img_val.py --mode feature
```
to get motion features of validation set

Run
```
python prepro_feats_train.py
```
to get visual features of train set
Run
```
python prepro_feats_val.py
```
to get visual features of validation set


