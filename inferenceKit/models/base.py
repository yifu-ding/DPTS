from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple, Dict, List, Any
from copy import copy, deepcopy
from itertools import islice, cycle

import torch
from torch import nn
from transformers.generation import GenerationConfig

from .llm import BaseLargeLanguageModel
from .prm import BaseProcessRewardModel
from .generation import *
from ..utils import InferenceConfig, update_single_config
from .generation.cot import _agg_prm_min_max, _agg_prm_last_max,_agg_majority_vote,_agg_orm_vote,_agg_prm_min_vote,_agg_prm_last_vote, _extract_answer


COT_GENERATION_METHOD = {
    "vanilla": vanilla_generate_cot,
}

PRM_COT_GENERATION_METHOD = {
    'dpts': dpts_generate_cot, 
}

VOTING_METHOD = {
    "min_max": _agg_prm_min_max,
    "last_max": _agg_prm_last_max,
    "majority_vote": _agg_majority_vote,
    "min_vote": _agg_prm_min_vote,
    "last_vote": _agg_prm_last_vote,
}

class BaseInferenceModel(nn.Module):
    def __init__(
            self, 
            generation_model: BaseLargeLanguageModel, 
            reward_model: Optional[BaseProcessRewardModel] = None, 
            inference_config: Optional[InferenceConfig] = None, 
            **kwargs,
        ):
        super().__init__()
        self.device = kwargs.pop("device", torch.cuda.current_device)
        
        self.generation_model = generation_model
        self.reward_model = reward_model
        
        self.base_config = None
        self.inference_config = self._default_config(self.base_config)
        self.inference_config.update_config(inference_config.config, inference_config.llm_config, inference_config.prm_config)
        
    
    @torch.inference_mode()
    def generate(
            self, 
            query:str,
            **kwargs,
        ):
        inference_config = deepcopy(self.inference_config)
        self.inference_config.update_config(**kwargs)
        
        prompt = self.prompt_template(query, inference_config)
        inputs = self.pre_process(prompt, inference_config)
        cots, scores = self.generate_cot(inputs, inference_config)
        cot_str_list, ans_list = self.post_process(cots, inference_config)
        response = self.vote_cot(cot_str_list, ans_list, scores, inference_config)
        return response
        
    def generate_step(
            self, 
            inputs: torch.Tensor, 
            inference_config: InferenceConfig, 
            **kwargs,
        ):
        config = inference_config.llm_config
        # config, _, _ = update_single_config(config, copy=True, **kwargs)
        if inference_config.config.step_method == "none":
            return self.generation_model.generate(inputs, config, **kwargs)
        else:
            raise NotImplementedError(f"Not implemention of step method {inference_config.config.step_method} !!!")
        
    def generate_cot(
            self, 
            inputs: torch.Tensor, 
            inference_config: InferenceConfig, 
        ) -> Tuple[List[Any], List[Any]]:
        
        config = inference_config.config
        
        if config.cot_method == "none":
            return self.generate_step(inputs, inference_config=inference_config)
        elif config.cot_method in COT_GENERATION_METHOD.keys():
            cot_generation_method = COT_GENERATION_METHOD[config.cot_method]
            return cot_generation_method(self, inputs, inference_config)
        elif config.cot_method in PRM_COT_GENERATION_METHOD.keys():
            assert self.reward_model != None, "Cot method {config.cot_method} need reward model!"
            prm_cot_generation_method = PRM_COT_GENERATION_METHOD[config.cot_method]
            return prm_cot_generation_method(self, inputs, inference_config)
        else:
            raise NotImplementedError(f"Not implemention of cot method {config.cot_method} !!!")

    def _default_config(self, base_config=None):
        llm_config = self.generation_model.config
        prm_config = self.reward_model.config if self.reward_model else None
        config = base_config or InferenceConfig(llm_config=llm_config, prm_config=prm_config)
        return config
    
    @abstractmethod
    def prompt_template(self, query:str, inference_config: InferenceConfig):
        raise NotImplementedError
    
    def pre_process(self, inputs, inference_config: InferenceConfig):
        return inputs
    
    def post_process(self, inputs, inference_config: InferenceConfig):
        return inputs
    
    def vote_cot(self, cots: List[str], ans_list: List[str], scores: List[List[float]], inference_config: InferenceConfig):
        return cots[0]
    

class DefaultInferenceModel(BaseInferenceModel):
    def __init__(self, generation_model, reward_model = None, inference_config = None, **kwargs):
        super().__init__(generation_model, reward_model, inference_config, **kwargs)
        
        update_single_config(self.inference_config.config, copy=False, **kwargs)

    def _default_config(self, base_config=None):
        config = super()._default_config(base_config)
        
        config.config.tree_width = 4
        config.config.tree_depth = 20
        
        config.config.step_method = "none"
        config.config.cot_method = "none"
        config.config.voting_method = "all"
        
        config.config.max_new_tokens = 2048
        # mcts
        config.config.max_rollout = 10
        config.config.max_step_time = 60
        return config
        
    def prompt_template(self, query: Union[str, List[str]], inference_config: InferenceConfig):
        message = lambda role, content: {"role": role, "content": content}

        system_prompt = """
You are an advanced mathematical assistant capable of solving problems across various areas of mathematics, \
including algebra, geometry, logic, calculus, and more. When solving any problem, \
carefully analyze the given information and select the appropriate methods or techniques for the solution. \
Break down the solution process into clear, logical steps, and explain each step in detail. \
Ensure that you provide any relevant definitions, formulas, or theorems where necessary. \
Avoid skipping steps, and ensure that the final answer is clearly stated at the end.
"""

        history_msgs = []
        history_msgs.append(message("system", system_prompt))
        
        if isinstance(query, str):
            history_msgs.append(message("user", query))
        elif isinstance(query, List[str]):
            for r, q in zip(cycle(["user", "assistant"]), query):
                history_msgs.append(message(r, q))
        
        chat = self.generation_model.tokenizer.apply_chat_template(
            history_msgs, add_generation_prompt=True, return_tensors='pt'
            ).to(self.device)
        
        return chat

    def post_process(self, inputs, inference_config):
        response_list = []
        answer_list = []
        
        for i in inputs:
            response = self.generation_model.tokenizer.batch_decode(
                i, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            ans = _extract_answer(response)
            response_list.append(response)
            answer_list.append(ans)
        return response_list, answer_list
    
    def vote_cot(self, cots: List[str], ans_list: List[str],scores: List[List[float]], inference_config: InferenceConfig) -> Dict[str, str]:
        """
        Aggregate multiple candidates using various strategies and return results as a dictionary.
        
        Parameters:
        - cots: List of candidate texts.
        - scores: List of score lists for each candidate.
        - ans_list: List of extracted answers for each candidate.

        Returns:
        A dictionary containing results from different aggregation strategies.
        """
        voting_method = inference_config.config.voting_method

        results = {}
        if voting_method in VOTING_METHOD:
            vote = VOTING_METHOD[voting_method]
            results[voting_method] = vote(x_list=cots, ans_list=ans_list, v_list=scores)
        elif voting_method == "all":
            for voting_method, vote in VOTING_METHOD.items():
                results[voting_method] = vote(x_list=cots, ans_list=ans_list, v_list=scores)
        else:
            raise NotImplementedError(f"Not implemention of voting method {voting_method} !!!")
        return results
    

class VLLMInferenceModel(BaseInferenceModel):
    def __init__(self, generation_model, reward_model = None, inference_config = None, **kwargs):
        super().__init__(generation_model, reward_model, inference_config, **kwargs)
        
        update_single_config(self.inference_config.config, copy=False, **kwargs)

    def _default_config(self, base_config=None):
        config = super()._default_config(base_config)
        
        config.config.tree_width = 4
        config.config.tree_depth = 20
        
        config.config.step_method = "none"
        config.config.cot_method = "none"
        config.config.voting_method = "all"
        
        config.config.max_new_tokens = 2048
        # mcts
        config.config.max_rollout = 10
        config.config.max_step_time = 60
        return config
        
    def prompt_template(self, query: Union[str, List[str]], inference_config: InferenceConfig):
        message = lambda role, content: {"role": role, "content": content}

        system_prompt = """
You are an advanced mathematical assistant capable of solving problems across various areas of mathematics, \
including algebra, geometry, logic, calculus, and more. When solving any problem, \
carefully analyze the given information and select the appropriate methods or techniques for the solution. \
Break down the solution process into clear, logical steps, and explain each step in detail. \
Ensure that you provide any relevant definitions, formulas, or theorems where necessary. \
Avoid skipping steps, and ensure that the final answer is clearly stated at the end.
"""

        history_msgs = []
        history_msgs.append(message("system", system_prompt))
        
        if isinstance(query, str):
            history_msgs.append(message("user", query))
        elif isinstance(query, List[str]):
            for r, q in zip(cycle(["user", "assistant"]), query):
                history_msgs.append(message(r, q))
        
        chat = self.generation_model.tokenizer.apply_chat_template(
            history_msgs, add_generation_prompt=True, tokenize=False
            )
        
        return chat

    def post_process(self, inputs, inference_config):
        response_list = []
        answer_list = []
        
        for response in inputs:
            ans = _extract_answer(response)
            response_list.append(response)
            answer_list.append(ans)
        return response_list, answer_list
    
    def vote_cot(self, cots: List[str], ans_list: List[str],scores: List[List[float]], inference_config: InferenceConfig) -> Dict[str, str]:
        """
        Aggregate multiple candidates using various strategies and return results as a dictionary.
        
        Parameters:
        - cots: List of candidate texts.
        - scores: List of score lists for each candidate.
        - ans_list: List of extracted answers for each candidate.

        Returns:
        A dictionary containing results from different aggregation strategies.
        """
        voting_method = inference_config.config.voting_method

        results = {}
        if voting_method in VOTING_METHOD:
            vote = VOTING_METHOD[voting_method]
            results[voting_method] = vote(x_list=cots, ans_list=ans_list, v_list=scores)
        elif voting_method == "all":
            for voting_method, vote in VOTING_METHOD.items():
                results[voting_method] = vote(x_list=cots, ans_list=ans_list, v_list=scores)
        else:
            raise NotImplementedError(f"Not implemention of voting method {voting_method} !!!")
        return results