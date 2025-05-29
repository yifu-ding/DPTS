import torch
import copy
import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import BaseProcessRewardModel, DefaultProcessRewardModel
from inferenceKit.utils import InferenceConfig


class MistralPRM(DefaultProcessRewardModel):
    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, **kwargs)
        
        self.good_token = self.config.good_token
        self.bad_token = self.config.bad_token
        
        self.step_token = self.config.step_token
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.candidate_tokens = self.tokenizer.encode(f"{self.good_token} {self.bad_token}")[1:]
        self.step_token_id = self.tokenizer.encode(f"{self.step_token}")[-1]
    
    def _default_config(self, base_config=None):
        config = super()._default_config(base_config)
        
        config.good_token = '+'
        config.bad_token = '-'
        
        config.step_token = ' ки'
        return config
    
    @torch.inference_mode()
    def score(self, query, inference_config: InferenceConfig):
        if not inference_config.config.mini_step:
            for step_token in inference_config.llm_config.step_tokens:
                query = list(map(lambda x:x.replace(step_token, step_token+inference_config.prm_config.step_token), copy.deepcopy(query)))
            
            input_ids = self.tokenizer(query, padding=True, return_tensors='pt')["input_ids"].to(self.device)
            logits = self.model(input_ids).logits[:,:, self.candidate_tokens]
            scores = logits.softmax(dim=-1)[:,:,0]
            step_scores = []
            for score, input_id in zip(scores.unbind(0), input_ids.unbind(0)):
                step_score = score[input_id == self.step_token_id]
                step_score = step_score.to(torch.float32).detach().cpu().numpy().tolist()
                step_scores.append(step_score)
            return step_scores
        else:
            query = list(map(lambda x:x+inference_config.prm_config.step_token, copy.deepcopy(query)))
            input_ids = self.tokenizer(query, padding=True, return_tensors='pt')["input_ids"].to(self.device)
            logits = self.model(input_ids).logits[:,:, self.candidate_tokens]
            scores = logits.softmax(dim=-1)[:,:,0]

            step_scores = []
            for score, input_id in zip(scores.unbind(0), input_ids.unbind(0)):
                step_score = score.to(torch.float32).detach().cpu().numpy().tolist()
                step_scores.append(step_score)
            return step_scores