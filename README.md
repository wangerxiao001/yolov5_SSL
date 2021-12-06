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
Smililar to the original yolov5, but with some adjustment as follows. the training file is [train_mt.py](https://github.com/wangerxiao001/yolov5_SSL/blob/c3f43a6778793bbe9c93a489b20af04d7eaa4ac1/train_mt.py)

### 2.1 Parameters added
