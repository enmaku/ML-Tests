from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from numpy import float64

import numpy as np
import random
import sys
import os
import re
import math
import itertools
from unidecode import unidecode

path = "datasets/pride_prejudice.txt"

text = unidecode(open(path, 'r', encoding='utf-8').read().lower())

print('corpus length:', len(text))

token_re = re.compile(r'(\w+|[\'])+|[.,!"]')  # TODO more punctuation and more exotic chars
# corpus_tokens should contain a list of tokens, which for the above regexp
# should be words and some punctuation marks
corpus_tokens = list(map(lambda match: match.string[match.start():match.end()],
                         token_re.finditer(open(path).read().lower())))
# Obtain a set of all unique tokens in the corpus
token_set = set(corpus_tokens)

print("total number of unique tokens", len(token_set))

# Read vectorizations
vectors_by_token = {}
embedding_dimensions = None
with open('datasets/pride_prejudice.vectors.txt') as f:
    for line in f:
        line = line.split()
        if embedding_dimensions is None:
            embedding_dimensions = len(line)-1
        elif embedding_dimensions != len(line)-1:
            print('each embedding vector needs the same number of dimensions')
            raise SystemExit(1)
        vectors_by_token[line[0]] = np.array(line[1:], np.float32)

# Ensure each token is represented by a vector
for token in token_set:
    if token not in vectors_by_token:
        print('token "%s" not found in vectors file' % token)
        raise SystemExit(1)

def vector_to_token(aVector):
    #print('aVector=', aVector)
    leastDistance = math.inf
    leastDistantToken = "NO TOKEN FOUND" # TODO use "<unk>" cause that's what standford glove uses
    # TODO linear search is dumb
    for bToken, bVector in itertools.islice(vectors_by_token.items(), 16):
        #print('bToken=', bToken)
        #print('bVector=', bVector)
        #print('aVector=', aVector)
        distance = np.linalg.norm(aVector - bVector)
        #print('distance=', distance)
        if distance < leastDistance:
            #print('new least distance', distance)
            #print('new nearest token', bToken)
            leastDistance = distance
            leastDistantToken = bToken
    return leastDistantToken

# the length of our input sequence, or, our feature vector length
seqlen = 30
# our stride in batching features from our corpus
step = 30
# the number of predictions we'll optimize for at a time
batch_size = 1024
# the number of times we call Keras' fit() method before demonstrating text
# generation
batches_per_demo = 32
print("seqlen:", seqlen, "step:", step)
# check that we'll get a healthy mix of sequences
if len(corpus_tokens) % step == 0:
    print('WARNING: your corpus is a multiple of your step size')

# NOTE unlike the previous version, we'll allocate tensors for a single
# batch, and repopulate them for each optimization
# training features tensor requires shape:
#   [batch size, sequence length, token encoding]
X = np.zeros((batch_size, seqlen, embedding_dimensions), dtype=np.float32)
# training labels tensor for predicting individual words requires shape:
#   [batch size, token encoding]
y = np.zeros((batch_size, embedding_dimensions), dtype=np.float32)


# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(seqlen, embedding_dimensions)))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(embedding_dimensions))
# model.add(Dense(1000))
model.add(Activation('softmax'))

model.compile(loss='mean_squared_error', optimizer='adam')

weights_filename = __file__ + '.weights.h5'
if os.path.isfile(weights_filename):
    model.load_weights(weights_filename)


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    a = a.astype(float64)
    a = a / a.sum(0)
    return np.argmax(np.random.multinomial(1, a, 1))


# we'll do our own batching to save on memory
batchCursor = 0

# train the model, output generated text after each iteration
for iteration in range(1, 300):
    print()
    print('-' * 50)
    print('Iteration', iteration)

    # populate our batch
    #sys.stdout.write('generating new training batch\n')
    for iSample in range(batch_size):
        #sys.stdout.write('batch #%d\n' % iSample)
        for iSeq, token in enumerate(corpus_tokens[batchCursor : batchCursor + seqlen]):
            #sys.stdout.write(token+' ')
            #print(token)
            X[iSample, iSeq] = vectors_by_token[token]
        #print('\nvectorization: ', X[iSample, iSeq])
        # one-hot for labels
        token = corpus_tokens[batchCursor + seqlen]
        y[iSample] = vectors_by_token[token]
        # increment batchCursor by step, wrapping around from
        # end-to-beginning, but respecting that there should be seqlen+1
        # additional tokens following the cursor
        batchCursor = (batchCursor + step) % (len(corpus_tokens) - seqlen - 1)

    start_index = random.randint(0, len(corpus_tokens) - seqlen - 1)

    if iteration % batches_per_demo == 0:
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)
            generated = ''
            sentence = corpus_tokens[start_index: start_index + seqlen]
            generated += ' '.join(sentence)
            print('----- Generating with seed: "', sentence, '"')
            print()
            sys.stdout.write(generated)
            print()

            for i in range(30):
                x = np.zeros((1, seqlen, embedding_dimensions))
                for t, token in enumerate(sentence):
                    x[0, t] = vectors_by_token[token]

                pred_token_vector = model.predict(x, verbose=0)[0]
                #next_token_vector = sample(preds, diversity)
                next_token = vector_to_token(pred_token_vector)
                generated += next_token
                del sentence[0]
                sentence.append(next_token)
                sys.stdout.write(' ')
                sys.stdout.write(next_token)
                sys.stdout.flush()
            print()

            break # TODO use diversity

    model.fit(X, y, batch_size=None, epochs=1)
    model.save_weights('lstm_character_gernation_weights.h5', overwrite=True)

    print('batchCursor = %s' % batchCursor)

model.save_weights('lstm_character_gernation_weights.h5')

# vim: set ts=4 sts=4 sw=4 :
