from vocab import Vocabulary
import evaluation
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
RUN_PATH = "coco_models/model_best_40L.pth.tar"
DATA_PATH = "/media/ubuntu/data/chunxiao/"
evaluation.evalvisual(RUN_PATH, data_path=DATA_PATH, split="test")
