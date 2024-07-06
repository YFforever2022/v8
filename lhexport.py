from ultralytics import YOLO
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml

import multiprocessing
import argparse

def parse_args():
    parse = argparse.ArgumentParser(description="你应该附带这些参数")
    parse.add_argument('--cfg', default="./1/cfg.yaml", help="默认为运行目录下的cfg.yaml文件")
    args = parse.parse_args()
    return args

if __name__ == '__main__':
    multiprocessing.freeze_support()
    args = parse_args()
    m_cfg = args.cfg

    m_overrides = yaml_load(check_yaml(m_cfg), append_filename=True)
    m_model_file = m_overrides['model']
    model = YOLO(m_model_file)

    results = model.export(cfg=m_cfg)
