#Here are some libraries you're likely to use. You might want/need others as well.
import re
import sys
from random import random
from math import log
from collections import defaultdict
import numpy as np

tri_counts=defaultdict(int) #counts of all trigrams in input
bi_counts=defaultdict(int) #counts of all bigrams in input

#this function currently does nothing.
def preprocess_line(line):
    line = re.sub(r'\d', '0', line)
    line = re.sub(r'\n', '', line) # removes newline character at the end of each line
    line = re.sub(r'[^A-Za-z.\s\d]', '', line)
    line = line.lower()
    line = '##' + line + '#'
    return line

def all_trigrams():
    trigrams = {}
    letters = [' ','#','.','0','a','b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    for l1 in letters:
        for l2 in letters:
            for l3 in letters:
                trigrams[l1+l2+l3]=0
    return trigrams

def calculate_mle_prob(tri_counts, bi_counts):
    mle_probs = all_trigrams()
    for key in tri_counts.keys():
        history = key[0:2]
        history_count = bi_counts[history]
        mle_probs[key] = tri_counts[key] / history_count
    print(mle_probs)
    return mle_probs

def optimize_alpha(model, val_data):
# only calculates perplexity on val data using add_alpha_prob from training data
    H_add_alpha = 0
    for i in range(len(val_data) - 2):
        trigram = val_data[i:i+3]
        H_add_alpha = H_add_alpha - log(model[trigram], 2)
    H_avg = H_add_alpha / (len(val_data)-2)
    perplexity = np.power(2, H_avg)
    return perplexity


def calculate_add_alpha_prob(tri_counts, bi_counts, alpha):
    add_alpha_probs = all_trigrams()
    for key in add_alpha_probs:
        history = key[0:2]
        history_count = bi_counts[history]
        if key in tri_counts.keys():
            add_alpha_probs[key] = (tri_counts[key] + alpha)  / (history_count + alpha*30)      # add alpha smoothing
        else:
            add_alpha_probs[key] = alpha / (history_count + alpha*30)
    return add_alpha_probs

def generate_from_LM(distribution):
    random_sequence = "#"
    for i in range(299):
        bigram = random_sequence[-2:]
        if bigram[-1] == '#':
            random_sequence = random_sequence + "#"
            bigram = random_sequence[-2:]
        random_sequence = random_sequence + append_char(bigram, distribution)

        #print(random_sequence)
    return random_sequence

def normalize_probs(probs):
    normalized_probs = []
    for prob in probs:
        normalized_probs.append(prob / sum(probs))
    return normalized_probs

def append_char(bigram, distribution):
    possible_chars = []
    probs = []
    for key in distribution.keys():
        if key[0:2] == bigram:
            possible_chars.append(key[-1])
            probs.append(distribution[key])
    #print(possible_chars)
    normalized_probs = normalize_probs(probs)
    random_list = np.random.choice(possible_chars, size=None, replace=True, p=normalized_probs)
    #print(random_list)
    random_char = random_list[0]
    return random_char

def get_br_en_distribution():
    br_en_distribution = {}
    with open('model-br.en') as f:
        for line in f:
            line = re.sub(r'\n', '', line)
            trigram = line[0:3]
            prob = float(line[4:])
            br_en_distribution[trigram] = prob
    return br_en_distribution

#here we make sure the user provides a training filename when
#calling this program, otherwise exit with a usage error.
if len(sys.argv) != 2:
    print("Usage: ", sys.argv[0], "<training_file>")
    sys.exit(1)

infile = sys.argv[1] #get input argument: the training file

#This bit of code gives an example of how you might extract trigram counts
#from a file, line by line. If you plan to use or modify this code,
#please ensure you understand what it is actually doing, especially at the
#beginning and end of each line. Depending on how you write the rest of
#your program, you may need to modify this code.
with open(infile) as f:
    tri_counts.clear()
    bi_counts.clear()
    for line in f:
        line = preprocess_line(line)
        #print(line)
        for j in range(len(line)-(2)):
            trigram = line[j:j+3]
            bigram = line[j:j+2]
            tri_counts[trigram] += 1
            bi_counts[bigram] += 1

#Some example code that prints out the counts. For small input files
#the counts are easy to look at but for larger files you can redirect
#to an output file (see Lab 1).
print("Trigram counts in ", infile, ", sorted alphabetically:")
for trigram in sorted(tri_counts.keys()):
    print(trigram, ": ", tri_counts[trigram])
print("Trigram counts in ", infile, ", sorted numerically:")
for tri_count in sorted(tri_counts.items(), key=lambda x:x[1], reverse = True):
    print(tri_count[0], ": ", str(tri_count[1]))

with open("val_en.rtf") as f:
    val_string = f.read()
    val_string = preprocess_line(val_string)

alpha_perplexity = {}
#alpha_perplexity[0] = optimize_alpha(calculate_mle_prob(tri_counts,bi_counts), val_string)
for i in range(1,20):
    alpha = i * 0.05
    model = calculate_add_alpha_prob(tri_counts,bi_counts,alpha)
    perplexity = optimize_alpha(model, val_string)
    alpha_perplexity[alpha] = perplexity
print(alpha_perplexity)

# some code to find min value in alpha_perp
best_alpha = 0.15

print(generate_from_LM(calculate_add_alpha_prob(tri_counts,bi_counts,best_alpha)))
print(generate_from_LM(get_br_en_distribution()))


# run this on test file
