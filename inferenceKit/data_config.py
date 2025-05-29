from functools import partial
from .dataset import *
import os
BENCHMARK_PATH= os.environ.get('BENCHMARK_ROOT',r'./benchmark') # r"./benchmark"
supported_dataset = {
    'math': partial(MathDataset, dataset_name='math', dataset_path=os.path.join(BENCHMARK_PATH,"math_test500_dataset.json")),
    'gsm8k': partial(GSM8KDataset, dataset_name='gsm8k', dataset_path=os.path.join(BENCHMARK_PATH,"gsm8k_test1319_dataset.json"))
}