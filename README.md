## Introduction
This is Bidirectional Focal Attention Network, source code of [BFAN](https://dl.acm.org/doi/10.1145/3343031.3350869) ([project page](https://github.com/CrossmodalGroup/BFAN)). The paper is accepted by ACMMM2019 as Oral Presentation. It is built on top of the [SCAN](https://github.com/kuanghuei/SCAN) in PyTorch.

## Requirements and Installation
We recommended the following dependencies.

* Python 2.7
* [PyTorch](http://pytorch.org/) 1.1.0
* [NumPy](http://www.numpy.org/) (>1.12.1)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)


## Download data
Download the dataset files. We use the dataset files created by SCAN [Kuang-Huei Lee](https://github.com/kuanghuei/SCAN). The word ids for each sentence is precomputed, and can be downloaded from [here](https://drive.google.com/open?id=1IoL1eJDQlaLDCub6zsmjDpAJDz7LjW59) (for Flickr30K and MSCOCO) 

## Training

```bash
python train.py --data_path "$DATA_PATH" --data_name coco_precomp --vocab_path "$VOCAB_PATH" --logger_name runs/log --max_violation 
```

Arguments used to train Flickr30K models and MSCOCO models are as same as those of SCAN:

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

Test on Flickr30K
```bash
python test.py
```

To do cross-validation on MSCOCO, pass `fold5=True` with a model trained using 
`--data_name coco_precomp`.

```bash
python testall.py
```

## Reference

If you found this code useful, please cite the following paper:
```
@inproceedings{liu2019focus,
  title={Focus Your Attention: A Bidirectional Focal Attention Network for Image-Text Matching},
  author={Liu, Chunxiao and Mao, Zhendong and Liu, An-An and Zhang, Tianzhu and Wang, Bin and Zhang, Yongdong},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia},
  pages={3--11},
  year={2019},
  organization={ACM}
}
```
