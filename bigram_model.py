# -*- coding: utf-8 -*-
"""
Created on Wed May 22 13:54:38 2024

@author: System150L3
"""

# This script 'trains' a bigram language model to create a new name

import torch

# get the list of all names
words = open('names.txt').read().splitlines()
total_names = len(words)

# get all the bigrams in the corpus
for name in words[:2]:
    for ch1, ch2 in zip(name, name[1:]):
        print(ch1, ch2)

