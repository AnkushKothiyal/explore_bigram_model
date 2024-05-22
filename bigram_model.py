# -*- coding: utf-8 -*-
"""
@author: Ankush
"""

# This script is a mode polished version of 'bigram_model_explored.py'. It only has the final code and no tinkering to the differnt parts of the process.
# Special thanks to Andrej Karpathy for his tutorials.


import torch
import matplotlib.pyplot as plt
import seaborn as sns

# get the list of all names
words = open('names.txt').read().splitlines()
total_names = len(words)

# Create a tensor to store the frequency for each bigram combination
N = torch.zeros((27,27), dtype = torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {char:index+1 for index,char in enumerate(chars)}
stoi['.'] = 0
itos = {value:key for key,value in stoi.items()}

for name in words:
    # Adding special Start and End characters/ tokens
    chs = ['.'] + list(name) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ch1_inx = stoi[ch1]
        ch2_inx = stoi[ch2]
        N[ch1_inx,ch2_inx] = N[ch1_inx,ch2_inx] + 1
all_tokens = sorted(list(itos.values()))

# Plot a heat map to visualize the bigram distributions
sns.heatmap(N, xticklabels=all_tokens, yticklabels=all_tokens, cmap="YlGnBu", annot=True, fmt='d')
plt.title("Bigram Matrix")
plt.xlabel("Second Letter")
plt.ylabel("First Letter")
plt.show()



# Create a torch generator to make sure the results are reproducible
g = torch.Generator().manual_seed(4444)
P = N.float()
P /= P.sum(dim=1, keepdims=True)

#Time for prediction
for i in range(10):
    out = []
    ix=0 #Starting with the inital token '.'
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, generator=g).item()
        out.append(itos[ix])
        if ix==0 and len(out)<=2:
            print(f"one letter name '- {''.join(out)}' predicted, reiterating :") #when it reaches the end token or if the name is just one letter
            out = out[:1]
            continue
        elif ix==0:
            break
    print(''.join(out))
