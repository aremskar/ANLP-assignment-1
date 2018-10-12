#Here are some libraries you're likely to use. You might want/need others as well.
import re
import sys
from random import random
from math import log
from collections import defaultdict
import numpy

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

print(preprocess_line('¿Sería apropiado que usted, Señora Presidenta, escribiese una carta'))

def calculate_mle_prob(tri_counts, bi_counts):
    mle_probs = {}
    H_ml = 0
    for key in tri_counts.keys():
        history = key[0:2]
        history_count = bi_counts[history]
        mle_probs[key] = tri_counts[key] / history_count
        H_ml -= log(mle_probs[key],2)
    H_avg = H_ml / len(tri_counts)
    pp_ml = np.power(2, H_avg)
    return mle_probs, pp_ml

def calculate_add_alpha_prob(tri_counts, bi_counts):
    add_alpha_probs = {}
    alpha = 0.5
    H_add_alpha = 0
    print('alpha is {}'.format(alpha))
    for key in tri_counts.keys():
        history = key[0:2]
        history_count = bi_counts[history]
        add_alpha_probs[key] = (tri_counts[key] + alpha)  / (history_count + alpha*30)
        H_add_alpha -= log(add_alpha_probs[key], 2)
    H_avg = H_add_alpha / len(tri_counts)
    pp_add_alpha = np.power(2, H_avg)
    return add_alpha_probs, pp_add_alpha

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
    random_list = numpy.random.choice(possible_chars, size=None, replace=True, p=normalized_probs)
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
            # if j == 0:
            #     trigram = '#'+line[j:j+2]
            #     bigram = '#' + line[j:j+1]
            #     tri_counts[trigram] += 1
            #     bi_counts[bigram] += 1

            trigram = line[j:j+3]
            bigram = line[j:j+2]
            tri_counts[trigram] += 1
            bi_counts[bigram] += 1

            # if j == len(line)-3:
            #     trigram = line[j+1:j+3] + '#'
            #     tri_counts[trigram] += 1
            #     bigram = line[j+1:j+3]
            #     bi_counts[bigram] += 1


#Some example code that prints out the counts. For small input files
#the counts are easy to look at but for larger files you can redirect
#to an output file (see Lab 1).
print("Trigram counts in ", infile, ", sorted alphabetically:")
for trigram in sorted(tri_counts.keys()):
    print(trigram, ": ", tri_counts[trigram])
print("Trigram counts in ", infile, ", sorted numerically:")
for tri_count in sorted(tri_counts.items(), key=lambda x:x[1], reverse = True):
    print(tri_count[0], ": ", str(tri_count[1]))


# mle_probs = calculate_mle_prob(tri_counts, bi_counts)
# print(generate_from_LM(mle_probs))
# br_en_dist = get_br_en_distribution()
# print(generate_from_LM(br_en_dist))
br_en_distribution = get_br_en_distribution()
print(generate_from_LM(br_en_distribution))
print(generate_from_LM(calculate_mle_prob(tri_counts, bi_counts)))
