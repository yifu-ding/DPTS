import argparse
import os
import datetime

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list

from inferenceKit import utils
from inferenceKit.model_config import create_llm, create_prm
from inferenceKit.data_config import supported_dataset
from inferenceKit.inference import inference
from inferenceKit.models import InferenceConfig, DefaultInferenceModel, VLLMInferenceModel

import torch.nn.functional as F
import random
import numpy as np
import time

def parse_args():
    parser = argparse.ArgumentParser()
    SUPPRESS = argparse.SUPPRESS
    
    parser.add_argument('-c', '--config', default=None, type=str, metavar='FILE', help='Json Config File specifying arguments')
    
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model',type=str, required=True)
    parser.add_argument('--model_path',type=str, default=None)
    parser.add_argument('--reward_model', type=str, default=None)
    parser.add_argument('--reward_model_path', type=str, default=None)
    
    parser.add_argument('--vllm', action='store_true', default=False, help='Enable vllm')
    
    parser.add_argument('--cot_method', type=str, default=SUPPRESS)
    parser.add_argument('--step_method', type=str, default=SUPPRESS)
    parser.add_argument('--voting_method', type=str, default=SUPPRESS)

    parser.add_argument('--max_step_time', type=int, default=SUPPRESS)

    parser.add_argument('--work-dir',type=str, default='./outputs')
    parser.add_argument('--exp-name', type=str, default=None)
    
    parser.add_argument('--dtype', type=str, default="float32")
    parser.add_argument('--flash-attn', action='store_true', default=False, help='Enable Flash Attention')
    parser.add_argument('--shard', action='store_true', default=False, help='Big Model Sharded Inference')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug Mode')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume work-dir Experiment')
    
    args = parser.parse_args()
    return args


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    seed_all()

    args = parse_args()
    args = vars(args)
    accelerator = Accelerator()
    accelerator.even_batches = False
    logger = utils.get_logger('MAIN')

    device_map = "auto" if args.pop("shard") or args["vllm"] else None
    dtype = getattr(torch, args.pop("dtype"))
    flash_attn = args.pop("flash_attn")

    config_file = args.pop("config", None)
    inference_config = InferenceConfig.from_json(config_file) if config_file else InferenceConfig()
    utils.update_single_config(inference_config.config, copy=False, **args)
    args = inference_config.config
    args.cot_method = args.cot_method.lower()
    print(args)
        
    reward_model = None
    if args.reward_model is not None:
        reward_model, args.reward_model_path = create_prm(args.reward_model, args.reward_model_path)
        reward_model = reward_model(model_path=args.reward_model_path, device=accelerator.device, device_map=device_map, dtype=dtype, flash_attn=flash_attn)
    
    generation_model, args.model_path = create_llm(args.model, args.model_path, args.vllm)
    generation_model = generation_model(model_path=args.model_path, device=accelerator.device, device_map=device_map, dtype=dtype, flash_attn=flash_attn)

    if args.vllm:
        model = VLLMInferenceModel(generation_model, reward_model, inference_config, device=accelerator.device)
    else:
        model = DefaultInferenceModel(generation_model, reward_model, inference_config, device=accelerator.device)
    model.eval()
    
    print("initialize dataloader")
    # initialize dataloader
    dataset = supported_dataset[args.data]()
    dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=1, collate_fn=lambda x:x[0])

    # distributed
    dataloader = accelerator.prepare(dataloader)
    
    print("initialize output directory")
    # initialize output directory
    final_inference_config = model.inference_config
    output_dir = None
    if accelerator.is_main_process:
        if args.resume:
            output_dir = args.resume
            resume_file = utils.get_resume_file(args.resume)
            resume_results = utils.collect_resume_results(resume_file)
        else:
            work_dir = utils.default_work_dir(args, final_inference_config)
            output_dir = utils.get_outdir(work_dir)
    output_dir = broadcast_object_list([output_dir], from_process=0)[0]
    
    if accelerator.is_main_process:
        final_inference_config.to_json(os.path.join(output_dir, "config.json"))
    
    print("inference")
    start_time = time.time()
    results = inference(model, dataloader, accelerator, output_dir)
    end_time = time.time()
    peak_memory = torch.cuda.max_memory_allocated()
    
    for k, res in results.items():
        all_results = accelerator.gather_for_metrics(res, True)
        
        if accelerator.is_main_process:
            all_results = sorted(all_results, key=lambda x: x["index"])
            utils.dump(all_results, os.path.join(output_dir, f"results-{k}.json"))
            
            dataset.evaluate_results(all_results)
    
    print("**** finish evaluate, config: ****")
    print(args)
    print(f"Overall inference time: {end_time - start_time:.3f} s, peak memory: {peak_memory / 1024 ** 3:.2f} GB")

if __name__ == '__main__':
    main()