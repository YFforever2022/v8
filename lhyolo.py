import requests
import os
import time
from ultralytics import YOLO
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml

import multiprocessing
import argparse

def parse_args():
    multiprocessing.freeze_support()
    gg()
    parse = argparse.ArgumentParser(description="你应该附带这些参数")
    parse.add_argument('--mode', default="train", help="默认为train，模式总共有train(训练)、predict(预测)、val(验证)")
    parse.add_argument('--cfg', default="./1/cfg.yaml", help="默认为主目录下的cfg.yaml文件")
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    multiprocessing.freeze_support()
    args = parse_args()
    m_cfg = args.cfg
    m_mode = args.mode

    m_overrides = yaml_load(check_yaml(m_cfg), append_filename=True)  # 读入cfg文件参数
    m_model_file = m_overrides['model']  # 获取到它的配置
    model = YOLO(m_model_file)

    if m_mode == "train":
        results = model.train(cfg=m_cfg)
    elif m_mode == "predict":
        source = m_overrides['source']
        results = model.predict(cfg=m_cfg, source=source)
    elif m_mode == "val":
        data = m_overrides['data']
        results = model.val(cfg=m_cfg, data=data)

