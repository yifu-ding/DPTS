import re
import regex
import multiprocessing
from math import isclose
from typing import Union
from collections import defaultdict

from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy
def is_polar_coordinate(expr: str) -> bool:
    polar_pattern = re.compile(r"\(\s*\d+(\.\d+)?\s*,\s*(pi|\d+(\.\d+)?|π|tau)\s*\)", re.IGNORECASE)
    return bool(polar_pattern.match(expr))

def clean_choice_answer(prediction: str) -> str:
    pred = prediction.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    choices = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    pred = choices[-1] if choices else pred.strip().strip(".")
    return pred.rstrip(".").rstrip("/")

def parse_numeric_value(value):
    value = regex.sub(",", "", str(value))
    try:
        return float(value)
    except:
        if value.endswith("%"):
            value = value.rstrip("%\\")
            try:
                return float(value) / 100
            except:
                pass
    return None

def is_numeric(value):
    return parse_numeric_value(value) is not None

def latex_to_pmatrix(input_string):
    input_string = input_string.strip()
    matrices = re.findall(r"\{.*,.*\}", input_string)
    pmatrix_list = [r"\begin{{pmatrix}}{}\end{{pmatrix}}".format(matrix.strip("{}").replace(",", r"\\")) for matrix in matrices]
    return ", ".join(pmatrix_list)

def simplify_fraction(expr):
    expr = re.sub(r"\\frac{(\d+)}{(\d+)}", r"(\1/\2)", expr)
    expr = re.sub(r"\\dfrac{(\d+)}{(\d+)}", r"(\1/\2)", expr)
    expr = re.sub(r"\\frac{(\d+)}(\d+)", r"(\1/\2)", expr)
    expr = re.sub(r"\\frac(\d+){(\d+)}", r"(\1/\2)", expr)
    expr = re.sub(r"\\frac(\d)(\d)", r"(\1/\2)", expr)

    return expr
def simplify_plus_minus(expr):
    expr = re.sub(r"(\\d+)\\\\pm(\\d+)\\\\sqrt{(\\d+)}", r"\1+\2*sqrt(\3),\1-\2*sqrt(\3)", expr)
    return expr

def math_equal(
    pred: Union[bool, float, str],
    ref: Union[float, str],
    allow_percentage: bool = True,
    allow_close_match: bool = True,
    check_timeout: bool = False,
) -> bool:
    if pred is None or ref is None:
        return False
    if str(pred).strip().lower() == str(ref).strip().lower():
        return True
    if ref in ["A", "B", "C", "D", "E"] and clean_choice_answer(pred) == ref:
        return True
    
    if "," in pred and "," in ref:
        if not (is_polar_coordinate(pred) or is_polar_coordinate(ref)):
            pred_parts = [x.strip() for x in pred.split(",")]
            ref_parts = [x.strip() for x in ref.split(",")]
            
            if set(pred_parts) == set(ref_parts):
                return True

    try:
        if is_numeric(pred) and is_numeric(ref):
            pred_num, ref_num = parse_numeric_value(pred), parse_numeric_value(ref)
            comparisons = [ref_num / 100, ref_num, ref_num * 100] if allow_percentage else [ref_num]
            return any(
                (numeric_equal(pred_num, comp) if allow_close_match else comp == pred_num)
                for comp in comparisons
            )
    except:
        pass

    if not pred and pred not in [0, False]:
        return False

    ref, pred = str(ref).strip(), str(pred).strip()

    ref, pred = simplify_fraction(ref), simplify_fraction(pred)
    ref, pred = simplify_plus_minus(ref), simplify_plus_minus(pred)

    if "pmatrix" in pred and "pmatrix" not in ref:
        ref = latex_to_pmatrix(ref)

    ref, pred = re.sub(r"[{}()\[\]]", "", ref), re.sub(r"[{}()\[\]]", "", pred)
    if pred.lower() == ref.lower():
        return True

    if (
        regex.match(r"(\(|\[).+(\)|\])", pred)
        and regex.match(r"(\(|\[).+(\)|\])", ref)
        and all(
            math_equal(part.strip(), ref_part.strip(), allow_percentage, allow_close_match)
            for part, ref_part in zip(pred[1:-1].split(","), ref[1:-1].split(","))
        )
    ):
        return True

    if (
        re.match(r"\\begin{(p|b)matrix}", pred)
        and re.match(r"\\begin{(p|b)matrix}", ref)
        and all(
            all(
                math_equal(pred_item.strip(), ref_item.strip(), allow_percentage, allow_close_match)
                for pred_item, ref_item in zip(pred_row.split("&"), ref_row.split("&"))
            )
            for pred_row, ref_row in zip(
                pred.strip("\\begin{pmatrix}\\end{pmatrix}").split("\\\\"),
                ref.strip("\\begin{pmatrix}\\end{pmatrix}").split("\\\\")
            )
        )
    ):
        return True

    if (
        pred.count("=") == 1 and ref.count("=") == 1
        and (symbolic_equal(f"{pred.split('=')[0].strip()} - ({pred.split('=')[1].strip()})",
                            f"{ref.split('=')[0].strip()} - ({ref.split('=')[1].strip()})")
             or symbolic_equal(f"-({pred})", ref))
    ):
        return True

    return call_with_timeout(symbolic_equal_process, pred, ref) if check_timeout else symbolic_equal(pred, ref)

def numeric_equal(a: float, b: float, tolerance=1e-4) -> bool:
    """
    Check if two floating-point numbers are approximately equal.
    """
    return isclose(a, b, rel_tol=tolerance)

def symbolic_equal(a_expr, b_expr) -> bool:
    """
    Check if two symbolic expressions are equivalent using sympy.
    """
    def try_parsing(expr):
        for parser in [parse_latex, parse_expr, latex2sympy]:
            try:
                return parser(expr.replace("\\\\", "\\"))
            except:
                continue
        return expr

    a, b = try_parsing(a_expr), try_parsing(b_expr)

    try:
        if str(a) == str(b) or a == b:
            return True
    except:
        pass

    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except:
        pass

    try:
        return abs(a.lhs - a.rhs).equals(abs(b.lhs - b.rhs))
    except:
        pass

    try:
        return numeric_equal(float(N(a)), float(N(b)))
    except:
        pass

    try:
        if a.shape == b.shape and a.applyfunc(lambda x: round(x, 3)).equals(b.applyfunc(lambda x: round(x, 3))):
            return True
    except:
        pass

    return False

def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)

def call_with_timeout(func, *args, timeout=1, **kwargs):
    output_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=func, args=args + (output_queue,), kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return output_queue.get()

def _test_math_equal():
    gt = r"\frac1{6}"
    pred = r"1/6"
    print(math_equal(pred, gt))

if __name__ == "__main__":
    _test_math_equal() 
