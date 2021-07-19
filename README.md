
# SLA

Official implementation of our the paper: 

**Sparse Label Assignment for Oriented Object Detection in Aerial Images** .

<div align=center><img width="540" height="420" src="output/demo.png"/></div>



## Usage
The code is very simple, just modify the parameters of the relevant shell files and then run it.

### Installation
```
conda create -n sla python=3.6 -y
source activate sla
conda install pytorch=1.5 torchvision cudatoolkit=10.1 -c pytorch

pip install -r requirement.txt

sh compile.sh
```

### Inference
```
sh demo.sh
```

### Training
1. Put the dataset to the `data` directory.
2. Prepare dateset via:
```
sh prepare.sh
```
3. Start training:
```
sh train.sh
```

### Evaluation
Conduct evaluation via:
```
sh test.sh
```


## Citation

If you find our work or code useful in your research, please consider citing:

```
@article{ming2021sparse,
  title={Sparse Label Assignment for Oriented Object Detection in Aerial Images},
  author={Ming, Qi and Miao, Lingjuan and Zhou, Zhiqiang and Song, Junjie and Yang, Xue},
  journal={Remote Sensing},
  volume={13},
  number={14},
  pages={2664},
  year={2021},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
