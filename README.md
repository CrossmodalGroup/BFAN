## Introduction
This is Bidirectional Focal Attention Network, source code of [BFAN](https://arxiv.org/abs/1909.11416) ([project page](https://github.com/chunxiaoliu6/BFAN)). The paper is accepted by ACMMM2019. It is built on top of the [SCAN](https://github.com/kuanghuei/SCAN) in PyTorch.

## Requirements and Installation
We recommended the following dependencies.

* Python 2.7
* [PyTorch](http://pytorch.org/) 1.1.0
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)


## Download data
Download the dataset files. We use the dataset files created by SCAN [Kuang-Huei Lee](https://github.com/kuanghuei/SCAN).

## Training

```bash
python train.py --data_path "$DATA_PATH" --data_name coco_precomp --vocab_path "$VOCAB_PATH" --logger_name runs/log --max_violation 
```

Arguments used to train Flickr30K models and MS-COCO models are as same as those of SCAN:

For Flickr30K:

| Method      | Arguments |
| :---------: | :-------: |
|  BFAN-equal   | `--max_violation --lambda_softmax=20 --focal_type=equal --num_epoches=15 --lr_update=15 --learning_rate=.0002 --embed_size=1024 --batch_size=128 `|
|  BFAN-prob    | `--max_violation --lambda_softmax=20 --focal_type=prob --num_epoches=15 --lr_update=15 --learning_rate=.0002 --embed_size=1024 --batch_size=128 `|

For MSCOCO:

| Method      | Arguments |
| :---------: | :-------: |
|  BFAN-equal   | `--max_violation --lambda_softmax=20 --focal_type=equal --num_epoches=20 --lr_update=15 --learning_rate=.0005 --embed_size=1024 --batch_size=128 `|
| BFAN-prob     | `--max_violation --lambda_softmax=20 --focal_type=prob --num_epoches=20 --lr_update=15 --learning_rate=.0005 --embed_size=1024 --batch_size=128 `|

## Evaluation

```bash
python test.py
```