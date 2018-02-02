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

# Assign each token an ordinal. Generate two-way mapping
ordinals_by_token = dict((c, i) for i, c in enumerate(token_set))
tokens_by_ordinal = dict((i, c) for i, c in enumerate(token_set))
# Map our corpus' tokens to their ordinals
corpus_ordinals = [ordinals_by_token[token] for token in corpus_tokens]
# TODO del corpus_tokens ?

# the length of our input sequence, or, our feature vector length
seqlen = 30
# our stride in batching features from our corpus
step = 30
# the number of predictions we'll optimize for at a time
batch_size = 128
# the number of times we call Keras' fit() method before demonstrating text
# generation
batches_per_demo = 8
print("seqlen:", seqlen, "step:", step)
# check that we'll get a healthy mix of sequences
if len(corpus_ordinals) % step == 0:
    print('WARNING: your corpus is a multiple of your step size')

# we'll use one-hot encoding representation of tokens. this means a huge
# bool vector for each token representation of length len(token_set)
# NOTE unlike the previous version, we'll allocate tensors for a single
# batch, and repopulate them for each optimization
# training features tensor requires shape:
#   [batch size, sequence length, token encoding]
X = np.zeros((batch_size, seqlen, len(token_set)), dtype=np.bool)
# training labels tensor for predicting individual words requires shape:
#   [batch size, token encoding]
y = np.zeros((batch_size, len(token_set)), dtype=np.bool)


# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(seqlen, len(token_set))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(token_set)))
# model.add(Dense(1000))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

if os.path.isfile('lstm_character_gernation_weights.h5'):
    model.load_weights('lstm_character_gernation_weights.h5')


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
    X.fill(0) # should be reasonably performant
    y.fill(0)
    # TODO can this be optimized further?
    for iSample in range(batch_size):
        # one-hot for features
        for iSeq, ordinal in enumerate(corpus_ordinals[batchCursor : batchCursor + seqlen]):
            X[iSample, iSeq, ordinal] = 1
        # one-hot for labels
        ordinal = corpus_ordinals[batchCursor + seqlen]
        y[iSample, ordinal] = 1
        # increment batchCursor by step, wrapping around from
        # end-to-beginning, but respecting that there should be seqlen+1
        # additional tokens following the cursor
        batchCursor = (batchCursor + step + seqlen + 1) % len(corpus_tokens)

    model.fit(X, y, epochs=2)
    model.save_weights('lstm_character_gernation_weights.h5', overwrite=True)

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

            for i in range(1024):
                x = np.zeros((1, seqlen, len(token_set)))
                for t, word in enumerate(sentence):
                    x[0, t, ordinals_by_token[word]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_word = tokens_by_ordinal[next_index]
                generated += next_word
                del sentence[0]
                sentence.append(next_word)
                sys.stdout.write(' ')
                sys.stdout.write(next_word)
                sys.stdout.flush()
            print()

    print('batchCursor = %s' % batchCursor)

model.save_weights('lstm_character_gernation_weights.h5')

# vim: set ts=4 sts=4 sw=4 :
