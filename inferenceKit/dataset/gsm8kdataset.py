from ..utils import load
from .basedataset import BaseDataset

class GSM8KDataset(BaseDataset):
    def load_data(self, dataset_path, **kwargs):
        dataset = load(dataset_path)
        
        data_list = []
        for idx, data in enumerate(dataset):
            if 'index' not in data:
                raise ValueError(f"Not found index in dataset {dataset_path} !!!")
            
            index = data['index']
            prompt = data['input'] 
            answer = data['label'] 
            
            query = self.construct_instruction(prompt)
            dic = {
                'index': index,
                'input': query,
                'answer': answer
            }
            data_list.append(dic)
            
        return data_list
    
    def construct_instruction(self, prompt):
        # GSM8K specific instruction format
        return f"Please reason step by step, and ensure that the final answer includes the correct unit (e.g., ^\circ for degrees if it’s an angle). Put your final answer within \\boxed{{}}: {prompt}" 

