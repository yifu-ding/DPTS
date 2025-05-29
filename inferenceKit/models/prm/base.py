from typing import Dict, Optional, Tuple
from abc import ABC, abstractmethod

import torch
from torch import nn
from transformers.generation import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from inferenceKit.utils import update_single_config

class BaseProcessRewardModel(nn.Module):
    def __init__(self, model_path, **kwargs):
        super().__init__()
        self.device = kwargs.pop("device", torch.cuda.current_device)
        self.device_map = kwargs.pop("device_map", None) or self.device
        
        self.base_config = self._load_config(model_path)
        self.config = self._default_config(self.base_config)
        self.config, _, _ = update_single_config(self.config, copy=True, **kwargs)
        
        self.model_path = model_path
        self.model, self.tokenizer = self._load_model(model_path)
    
    def _load_config(self, model_path):
        try:
            config = GenerationConfig.from_pretrained(model_path)
        except:
            config = GenerationConfig()
        return config
    
    def _load_model(self, model_path):
        if model_path != None:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map=self.device_map, 
                torch_dtype=self.config.dtype, 
                attn_implementation="flash_attention_2" if self.config.flash_attn else None,
                )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    
    def _default_config(self, base_config=None):
        config = base_config or GenerationConfig()
        return config
    
    @abstractmethod
    @torch.inference_mode()
    def score(self, query, inference_config):
        raise NotImplementedError


class DefaultProcessRewardModel(BaseProcessRewardModel):
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, **kwargs)
    
    @torch.inference_mode()
    def score(self, query, inference_config):
        inputs = self.tokenizer(query, return_tensors="pt")
        for k,v in inputs.items():
            inputs[k] = v.to(self.device)
        return self.model(**inputs).logit