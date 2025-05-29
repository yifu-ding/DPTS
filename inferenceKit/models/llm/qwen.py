from .base import BaseLargeLanguageModel, DefaultLargeLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import warnings

class QwenLLM(BaseLargeLanguageModel):
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, **kwargs)
        
        self.step_tokens = self.config.step_tokens
        self.step_token_ids = self.tokenizer.encode(self.step_tokens)
        self.config.step_token_ids = self.step_token_ids
    
    def _default_config(self, base_config=None):
        config = super()._default_config(base_config)
        config.step_tokens = ["]\n\n", ".\n\n"]
        return config 
    
