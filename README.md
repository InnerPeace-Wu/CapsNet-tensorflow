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

* clone the repo, and set up parameters in `code/config.py`
* then 

```bash
cd $ROOT/code
python train.py
```
or train with logs by runing:(not tested yet)
```bash
$ bash train.sh
```

## Experiments

Note: all trained with `batch_size = 100`

latest commit with `3 iterations of dynamic routing`:
    
    1. update dynamic routing with tf.while_loop and static way
    2. fix margin loss issue
    
**result:**

Iterations | 1k     | 2k    | 3k    | 4k    | 5k    
:---------:|:------:|:-----:|:-----:|:-----:|:-----:
  val_acc  | 98.90  | 99.16 | 99.09 | 99.30 | - 
  test_acc |   -    |   -   | -     |   -   |   -   

commit [8e3785d](https://github.com/InnerPeace-Wu/CapsNet-tensorflow/tree/8e3785d5b6f34c13c81555edd97a6241a7885209). 

    with bugs:
    1. wrong implementation of margin loss
    2. updating `prior` during routing 
    
**result:**

Iterations | 2k     | 4k    | 5k    | 7k    | 9k    | 10k   
:---------:|:------:|:-----:|:-----:|:-----:|:-----:|:-----:
  val_acc  | 98.02  | 98.58 |  -    | 98.82 | 98.96 | -
  test_acc |   -    |   -   | 98.89 |   -   |   -   | 99.09 
  



## TODO
- [ ] fix the inefficacy
- [ ] report exclusive experiment results

## Reference

* [Keras implementation](https://github.com/XifengGuo/CapsNet-Keras)
* Discussion about **routing algorithm**, [issue](https://github.com/naturomics/CapsNet-Tensorflow/issues/8) and [issue](https://github.com/XifengGuo/CapsNet-Keras/issues/1)
