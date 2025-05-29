import pandas as pd
from abc import abstractmethod
import json
from sympy import simplify
from sympy.parsing.latex import parse_latex
import re
from  tqdm import tqdm, trange

from torch.utils.data import Dataset


from .grader import *

def clean_latex(s):
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    s = re.sub(r'\\left|\right', '', s)
    s = re.sub(r'\s+', '', s)
    s = re.sub(r'\\begin\{pmatrix\}|\\end\{pmatrix\}', '', s)
    return s

class BaseDataset(Dataset):
    def __init__(self, dataset_name, dataset_path, **kwargs):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.data = self.load_data(dataset_path, **kwargs)
    
    def load_data(self, dataset_path, **kwargs):
        pass
    
    def __getitem__(self, idx):
        return dict(self.data[idx])
    
    def __len__(self):
        return len(self.data)

    def evaluate(self, result_file, print_result=False, print_no_match=False):
        '''
        example:
        dataset = BaseDataset("", "")
        result = dataset.evaluate("math.json",print_result=True,print_no_match=True)
        '''
        with open(result_file, 'r', encoding="utf-8") as f:
            results = json.load(f)
        self.evaluate_results(results, print_result, print_no_match)


    def evaluate_results(self, results, print_result=False, print_no_match=False):
        total_samples = 0
        correct_samples = 0
        no_match_samples = 0
        print("Evaluating...\n")
        for idx, sample in enumerate(tqdm(results, desc="Submitting tasks to executor", unit="sample")):
            total_samples += 1
            response_list = str(sample.get('response', []))
            answer = str(sample.get('answer', ''))
            final_response = response_list if response_list else ""
            boxed_match = re.findall(r'\\boxed{((?:[^{}]|{[^{}]*})*)}', final_response)
            if not boxed_match or all(match == '' for match in boxed_match):
                no_match_samples += 1

                if print_no_match:
                    print("-------------------------------------------------")
                    print("response_is:",response_list)
                    print("true answer is:",answer)
                    print("-------------------------------------------------")
                continue


            for final_response in boxed_match:
                final_response = clean_latex(final_response)
                answer = clean_latex(answer)      
                if math_equal(final_response, answer):
                    correct_samples += 1
                    break

                else:
                    if print_result:
                        if final_response!='':
                            print("***********************")
                            print("final_response is: \n", final_response)
                            print("answer is: \n", answer)

        
        accuracy = correct_samples / total_samples if total_samples > 0 else 0.0

        evaluation_result = {
            "accuracy": accuracy,
            "total_samples": total_samples,
            "correct_samples": correct_samples,
            "no_match_samples": no_match_samples
        }
        
        print("Evaluation Result:", evaluation_result)
        print("no_match_samples is:", no_match_samples)
        return evaluation_result

    def compare(self, result_file1: str,result_file2: str):
        """
        Compare two json files and return the results.
        
        output:compare_result{
        a_correct_b_incorrect,
        a_incorrect_b_correct,
        a_incorrect_b_incorrect 
        }

        example:
        compare_result=dataset.compare("inferenceKit\dataset\math.json","inferenceKit\dataset\math_none.json")
        print('a_correct_b_incorrect:',compare_result["a_correct_b_incorrect"])
        print('a_incorrect_b_correct:',compare_result["a_incorrect_b_correct"])
        print('a_incorrect_b_incorrect:',compare_result["a_incorrect_b_incorrect"])
        """
        with open(result_file1, 'r') as file1:
            data1 = json.load(file1)
        with open(result_file2, 'r') as file2:
            data2 = json.load(file2)
        a_correct_b_incorrect = []
        a_incorrect_b_correct = []
        a_incorrect_b_incorrect = []
        for item1, item2 in zip(data1, data2):
            answer = str(item1.get('answer', ''))
            a_response = str(item1.get('response', []))[-100:]
            b_response = str(item2.get('response', []))[-100:]

            a_response = re.search(r'\\boxed{((?:(?!text{)(?:[^{}]|{[^{}]*}))*?)}', a_response)
            a_final_response=clean_latex(a_final_response)
            b_response = re.search(r'\\boxed{((?:(?!text{)(?:[^{}]|{[^{}]*}))*?)}', b_response)
            b_final_response=clean_latex(b_final_response) 
            answer=clean_latex(answer)

            a_correct = math_equal(a_final_response, answer)
            b_correct = math_equal(b_final_response, answer)

            if a_correct and not b_correct:
                a_correct_b_incorrect.append({
                    'answer': answer,
                    'A_response': a_final_response,
                    'B_response': b_final_response
                })
            elif not a_correct and b_correct:
                a_incorrect_b_correct.append({
                    'answer': answer,
                    'A_response': a_final_response,
                    'B_response': b_final_response
                })
            elif not a_correct and not b_correct:
                a_incorrect_b_incorrect.append({
                    'answer': answer,
                    'A_response': a_final_response,
                    'B_response': b_final_response
                })

            compare_result = {
                    "a_correct_b_incorrect":a_correct_b_incorrect,
                    "a_incorrect_b_correct":a_incorrect_b_correct,
                    "a_incorrect_b_incorrect":a_incorrect_b_incorrect,
                }

        return compare_result




