# -*- coding: utf-8 -*-
"""
@author: Ankush
"""

# This script builds the understanding in bigrams model training by twinkering with multiple aspects of it. 
#For most parts it starts with a non-effcient solution and then builds up to the efficient code.
# Special thanks to Andrej Karpathy for his tutorials.

"""
 A bigram model simply samples out of the distribution of all the bigrams in the training data.
 So, if we want to predict the next word in 'Emil' it will look at the bigram distribution of all pairs
 starting with the letter 'l' and will sample from that distribution.
""" 

import torch
import matplotlib.pyplot as plt
import seaborn as sns

# get the list of all names
words = open('names.txt').read().splitlines()
total_names = len(words)

# get all the bigrams in the corpus
for name in words[:2]:
    for ch1, ch2 in zip(name, name[1:]):
        print(ch1, ch2)

"""
 The issue with the above code is that we are not capturing the distribution of characters on which a name starts or ends.
 for e.g. for the name emma we are not considering that it is starting with 'e' and ending with 'a'
 To account for the ending and starting characters we'll add special start and end tokens <S> and <E>
"""

for name in words[:2]:
    # Adding special Start and End characters/ tokens
    chs = ['<S>'] + list(name) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        print(ch1, ch2)

"""
Now, for creating the distribution of all of these bigrams we'll count the instance of each of them. First using a dictionary and then using a tensor
"""       

#Using Python dictionary to count the instance of each of the bigrams
bigram_counts = {}
for name in words:
    # Adding special Start and End characters/ tokens
    chs = ['<S>'] + list(name) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
sorted(bigram_counts.items(), key = lambda kv: kv[1], reverse=True)

#Using torch tensor to store the counts
N = torch.zeros((28,28), dtype = torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {char:index for index,char in enumerate(chars)}
stoi['<S>'] = 26
stoi['<E>'] = 27
itos = {value:key for key,value in stoi.items()}

for name in words:
    # Adding special Start and End characters/ tokens
    chs = ['<S>'] + list(name) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        ch1_inx = stoi[ch1]
        ch2_inx = stoi[ch2]
        N[ch1_inx,ch2_inx] = N[ch1_inx,ch2_inx] + 1
all_tokens = list(itos.values())

# Plot a heat map to visualize the bigram distributions
sns.heatmap(N, xticklabels=all_tokens, yticklabels=all_tokens, cmap="YlGnBu", annot=True, fmt='d')
plt.title("Bigram Matrix")
plt.xlabel("Second Letter")
plt.ylabel("First Letter")
plt.show()

"""
Two things that are worth noting in the heatmap are:
1. Since no bigram would end with '<S>' therefore the column wrt to '<S>' is all zeros.(N[:,26])
2. Since no bigram would start with '<E> therefore the row wrt to '<E>' is all zeros.(N[27,:])
Thus, our tensor contains two reduntant dimensions (one row and one column).
In our case we can use a single special token to denote both start and end of a name and we wont lose on any information.
"""

# Replacing '<S>' and '<E>' with '.'
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

"""
Now, each row of the tensor represents the bigram distribution for that letter. As the model encounters a specific character as the last character in a string while predicting
it will look at the row corresponding to that character and samples the from that row.
To do so, each row needs to be normalised to convert the frequencies into probabilities.
"""

# Create a torch generator to make sure the results are reproducible
g = torch.Generator().manual_seed(4444)
P = N.float()
P /= P.sum(dim=1, keepdims=True)

# ix = torch.multinomial(p, num_samples=1, generator=g).item()

#Time for prediction
for i in range(5):
    out = []
    ix=0 #Starting a new name - hence starting with '.'
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, generator=g).item()
        out.append(itos[ix])
        if ix==0: #when it reaches the end token
            break
    print(''.join(out))
    
    

    



