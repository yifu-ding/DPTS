from .models import *
from . import *
from functools import partial

########## llm model

supported_SFT_LLM = {
    'llama-8b': (ABC, 'meta-llama/Llama-3.1-8B-Instruct'),
    'llama-1b': (ABC, 'meta-llama/Llama-3.2-1B-Instruct'),
    'llama-3b': (ABC, 'meta-llama/Llama-3.2-3B-Instruct'),
    'qwen-1.5b': (QwenLLM, 'Qwen/Qwen2.5-1.5B-Instruct'),
    'qwen-7b': (QwenLLM,  "Qwen/Qwen2.5-7B-Instruct"),
    'qwen-14b': (QwenLLM,  "Qwen/Qwen2.5-14B-Instruct"),
    'qwen-math-1.5b': (QwenLLM, 'Qwen/Qwen2.5-Math-1.5B-Instruct'),
    'qwen-math-7b': (QwenLLM,  "Qwen/Qwen2.5-Math-7B-Instruct"),
}    

######### reward model

supported_Reward_LLM = {
    'mistral_prm-7b': (MistralPRM, 'peiyi9979/math-shepherd-mistral-7b-prm')
}

def create_llm(model_name, model_path=None, vllm=False):
    llm_class, default_path = supported_SFT_LLM[model_name]
    model_path = model_path or default_path
    base_class = VLLMLargeLanguageModel if vllm else DefaultLargeLanguageModel
    fin_llm_class = type("DynamicLLM", (llm_class, base_class), {})
    return fin_llm_class, model_path

def create_prm(model_name, model_path=None):
    llm_class, default_path = supported_Reward_LLM[model_name]
    model_path = model_path or default_path
    return llm_class, model_path