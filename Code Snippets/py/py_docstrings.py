# -*- coding: utf-8 -*-
"""
@author: s.jayanthi
"""

# Docstrings Usage
import numpy as np

def funcA(arg1: int, arg2: list, arg3: 'string' = 'My Default String')->float:
    """
    Python does not automatically check if your inputs are oif same type as required by this method. You have to check it yourself. Source: https://stackoverflow.com/questions/2489669/function-parameter-types-in-python
    """
    if not isinstance(arg1, int):
        raise TypeError
    if not isinstance(arg2, list):
        raise TypeError('arg2: list')
    print(arg3)
    return np.float(arg1)
help(funcA)
print(funcA.__annotations__)
print(funcA(10.34324,['hello','bye']))   # ----> TypeError thrown
print(funcA(10,'Alpha'))                 # ----> TypeError thrown
print(funcA(10,['hello','bye']))         # ----> No error thrown
print(funcA(10,[],'Alpha'))              # ----> No error thrown

ret_dict = {'type': list, 'contains': 'a modified list', 'docstring':'A sample method is coded'}
def funcB(arg1: 'a number', arg2: 'a list', arg3: 'a string' = 'My Default String')->ret_dict:
    if not isinstance(arg1, int):
        raise TypeError
    if not isinstance(arg2, list):
        raise TypeError('arg2: list')
    if arg1>len(arg2):
        raise Exception('arg1<=len(arg2)')
    print('Before change: ', arg2)
    arg2[arg1-1] = arg3;
    print('After change: ', arg2)
    return arg2
help(funcB)
print(funcB.__annotations__)
print(funcB(10,['hello','bye']))   # ----> Exception thrown
print(funcB(1,['hello','bye']))    # ----> No error thrown