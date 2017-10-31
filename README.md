# Dynamic Routing Between Capsules
reference: [Dynamic routing between capsules](https://arxiv.org/abs/1710.09829v1) by **Sara Sabour, Nicholas Frosst, Geoffrey E Hinton**

Note: this implementation strictly follow the instructions of the paper, check the paper for details.

## Dependencies

* Codes are tested on `tensorflow 1.3`, and `python 2.7`. But it should be compatible with `python 3.x`
* Other dependencies as follows, install it by running `pip install -r requirements.txt` in `ROOT` directory.

```
numpy>=1.7.1
scipy>=0.13.2
easydict>=1.6
tqdm>=4.17.1
```

## Train

* clone the repo
* then 

```bash
cd $ROOT
python train.py
```

NOTE: First try with `50` iterations, it got `69.91%` accuracy on test set.

## TODO
- [ ] report exclusive experiment results
