# -*- coding: utf-8 -*-
"""
@author: s.jayanthi
"""

# Input Arguments

def funcA(arg1: 'a string', *multiple_args)->None:
    for arg in multiple_args:
        print(arg)
help(funcA)
funcA('Hello',5,[2,3,4,5],{2:'3',4:'5'})

def funcB(arg1: 'a string', **key_word_args)->None:
    for key, name in key_word_args.items():
        print(tuple((key,name)))
help(funcB)
funcB('Hello',arg2=5,arg3=[2,3,4,5],argn={2:'3',4:'5'});