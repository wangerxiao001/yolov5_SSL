# Usage of the newest branch dev2 of Yolov5_SSL
This rep is the implementation of semi-supervised learning with Yolov5 network, and it's mainly based on the mean-teacher SSL framework.

relative paper:

[mean-teacher](https://arxiv.org/abs/1703.01780)

[unbiased teacher](https://arxiv.org/abs/2102.09480)

## 1. Install
Similar to the [original install method of yolov5](https://github.com/ultralytics/yolov5)
```
git clone https://github.com/wangerxiao001/yolov5_SSL.git
git checkout dev2
pip install -r requirements.txt
cd yolov5_SSL
```

## 2. Usage
Smililar to the original yolov5, but with some adjustment as follows. We recommend you to read the code. the training file is [train_mt.py](https://github.com/wangerxiao001/yolov5_SSL/blob/c3f43a6778793bbe9c93a489b20af04d7eaa4ac1/train_mt.py)

### 2.1 Parameters added
- "--unsup-loss-weight":  Weight for unsupervised weight
- "--epochs": Number of total epochs, including burn-in epochs
- "--burnin-epochs": Warm-up epochs of mean-teacher training
- "--ema-rate": Weight for teacher model of EMA in mean-teacher
- "--conf_thresh-mt": confidence thresh for pseudo labels
- "-run-test": Run test after training

### 2.2 Config file
Similar to the [original config file of yolov5] (https://github.com/ultralytics/yolov5/blob/1075488d893f2167737d89549c3f675b0713aa5a/data/coco128.yaml), with the un-labeled data directory "train_u" added
```
# create by wangerxiao001
# date: 2021-11-29
# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: ./msc_ssl_data/images/train_p5/
val: ./msc_ssl_data/images/val_p5/
test: ./msc_ssl_data/images/test_p5/
train_u: ./msc_ssl_data/images_p8_p10/

# number of classes
nc: 2

# class name
names: ['Aging', 'Young']

# this is coco anchors, change it if necessary
anchors_scales: '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
anchors_ratios: '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'
```

### 2.3 Training
```
python train_mt.py --batch 12 --data data/msc_yolo_ssl_p5.yaml --weights yolov5s.pt --epochs 200 --burnin-epochs 100 --device 3 --ema-rate 0.99
```
