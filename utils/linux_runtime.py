"""Local Linux defaults; set these before importing torch/transformers."""
import os
import sys


def configure_environment():
    sys.dont_write_bytecode = True
    # cuda:0 is physical GPU 1; the existing runner uses one GPU per process.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
