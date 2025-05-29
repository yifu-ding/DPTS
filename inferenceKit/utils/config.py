from typing import Optional, Union, Tuple, Dict
from copy import deepcopy
import json

from transformers.generation import GenerationConfig
from vllm import SamplingParams

from .file import dump, load

def get_config_from_args(args):
    config = vars(args)
    config.pop('model', None)
    config.pop('reward_model', None)
    config.pop('config', None)
    return config

#TODO: receive a dict
def update_single_config(config: GenerationConfig, copy=False, **kwargs) -> Tuple[GenerationConfig, Dict, Dict]:
    if copy:
        config = deepcopy(config)
    updated_configs = {}
    added_configs = {}
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
            updated_configs[key] = value
        else:
            setattr(config, key, value)
            added_configs[key] = value
    # config.validate()
    return config, updated_configs, added_configs
    
def convert_to_str(obj):
    if isinstance(obj, dict):
        return {key: convert_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_str(item) for item in obj]
    elif isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj 
    else:
        return str(obj) 


class InferenceConfig:
    def __init__(
        self, 
        config: Optional[GenerationConfig] = None, 
        llm_config: Optional[GenerationConfig] = None, 
        prm_config: Optional[GenerationConfig] = None,
    ) -> None:
        
        self.config = config or GenerationConfig()
        self.llm_config = llm_config or GenerationConfig()
        self.prm_config = prm_config
    
    @classmethod
    def from_json(cls, path: str) -> 'InferenceConfig':
        config = load(path)
        for k, v in config.items():
            config[k] = GenerationConfig.from_dict(v) if v else None
        inference_config = InferenceConfig(**config)
        return inference_config
    
    def to_json(self, path: str):
        dic = {}
        
        dic["config"] = self.config.to_diff_dict()
        dic["llm_config"] = self.llm_config.to_diff_dict()
        if self.prm_config:
            dic["prm_config"] = self.prm_config.to_diff_dict()
        
        dic = convert_to_str(dic)
        dump(dic, path)
    
    def update_config(
        self, 
        config: Optional[GenerationConfig] = None, 
        llm_config: Optional[GenerationConfig] = None, 
        prm_config: Optional[GenerationConfig] = None,
        **kwargs,    
    ):
        gen_dict = kwargs
        if config != None:
            config, _, _ = update_single_config(config, copy=True, **gen_dict)
            gen_dict = config.to_diff_dict()
        update_single_config(self.config, **gen_dict)
        
        if llm_config != None:
            llm_dict = llm_config.to_diff_dict()
            update_single_config(self.llm_config, **llm_dict)

        if prm_config != None:
            prm_dict = prm_config.to_diff_dict()
            update_single_config(self.prm_config, **prm_dict)
        
        # gen_dict = {}
        # llm_dict = {}
        # prm_dict = {}
        
        # for key, value in kwargs:
        #     if key.startswith("llm_"):
        #         llm_dict[key[:4]] = value
        #     elif key.startswith("prm_"):
        #         prm_dict[key[:4]] = value
        #     else:
        #         gen_dict[key[:4]] = value
        
        # if config != None:
        #     config, _, _ = update_single_config(config, copy=True, **gen_dict)
        #     gen_dict = config.to_diff_dict()
        # update_single_config(self.config, **gen_dict)
        
        # if llm_config != None:
        #     llm_config, _, _ = update_single_config(llm_config, copy=True, **llm_dict)
        #     llm_dict = llm_config.to_diff_dict()
        # update_single_config(self.llm_config, **llm_dict)

        # if prm_config != None:
        #     prm_config, _, _ = update_single_config(prm_config, copy=True, **prm_dict)
        #     prm_dict = prm_config.to_diff_dict()
        # update_single_config(self.prm_config, **prm_dict)


def update_sampling_params_from_generation_config(samplingParams: SamplingParams, generation_config: Dict):
    key_diff_mapping = {"max_new_tokens": "max_tokens"}
    gen_dict = generation_config.to_diff_dict()
    for gen_key, samp_key in key_diff_mapping.items():
        gen_dict[samp_key] = gen_dict[gen_key]
        del gen_dict[gen_key]
    for k, v in gen_dict.items():
        setattr(samplingParams, k, v)
    return samplingParams