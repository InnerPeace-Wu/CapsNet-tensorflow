# Dynamic Routing Between Capsules
reference: [Dynamic routing between capsules](https://arxiv.org/abs/1710.09829v1) by **Sara Sabour, Nicholas Frosst, Geoffrey E Hinton**

Note: this implementation strictly follow the instructions of the paper, check the paper for details.

## Dependencies

* Codes are tested on `tensorflow 1.3`, and `python 2.7`. But it should be compatible with `python 3.x`
* Other dependencies as follows, 

```
numpy>=1.7.1
scipy>=0.13.2
easydict>=1.6
tqdm>=4.17.1
```
install by running 

```bash
$ pip install -r requirements.txt
``` 
in `ROOT` directory.



## Train

* clone the repo
* then 

```bash
cd $ROOT
python train.py
```

## Experiments

1. current commit. With several bugs about to fix.

Iterations | 2k     | 4k    | 5k    | 7k    | 9k    | 10k   
:---------:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:
  val_acc  | 98.02  | 98.58 |  -    | 98.82 | 98.96 | -
  test_acc |   -    |   -   | 98.89 |   -   |   -   | 99.09 


## TODO
- [ ] report exclusive experiment results
