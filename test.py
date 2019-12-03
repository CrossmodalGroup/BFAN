# -------------------------------------------------------------------------------------
# A Bidirectional Focal Atention Network implementation based on
# https://arxiv.org/abs/1909.11416.
# "Focus Your Atention: A Bidirectional Focal Atention Network for Image-Text Matching"
# Chunxiao Liu, Zhendong Mao, An-An Liu, Tianzhu Zhang, Bin Wang, Yongdong Zhang
#
# Writen by Chunxiao Liu, 2019
# -------------------------------------------------------------------------------------
"""test"""

from vocab import Vocabulary
import evaluation
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
RUN_PATH = "/userhome/BFAN/models/model_best.pth.tar"
DATA_PATH = "/userhome/"
evaluation.evalrank(RUN_PATH, data_path=DATA_PATH, split="testall",fold5=True)
