"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""

from abc import ABC, abstractmethod
import torch
from typing import Optional, Dict, Any, Tuple
from transformers.cache_utils import DynamicCache


class Node(ABC):
    cnt = 0
    def __init__(self, parent) -> None:
        super().__init__()

        # global node_cnt
        self.id = self.cnt
        self.__class__.cnt += 1

        self.parent = parent
        self.children = set()
        
        self.past_key_values = DynamicCache()

    def get_kv(self):
        return self.past_key_values.key_cache, self.past_key_values.value_cache
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        layer_idx: int = -1, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.key_cache is None: 
            self.key_cache = key_states
            self.value_cache = value_states 
        else:
            self.key_cache = torch.cat([self.key_cache, key_states], dim=-2)
            self.value_cache = torch.cat([self.value_cache, value_states], dim=-2)

        return self.key_cache, self.value_cache
    
    def get_seq_length_by_node_index(self) -> int:
        return self.key_cache.shape[-2]

    def __repr__(self):
        return f"Node_{self.id}"

class Searcher:
    def __init__(self):
        self.nodes = set()
        self.expanded_nodes = set()

    def process(self):
        return NotImplementedError
    
    def finalize(self):
        return NotImplementedError


def vanilla_generate_cot(self, inputs, generate_config):
    inputs = inputs
    
    while not self.cot_stop(inputs):
        step = self.generate_step(inputs, generate_config)
        inputs = torch.cat([inputs, step], dim=-1)
    output = inputs
    return output