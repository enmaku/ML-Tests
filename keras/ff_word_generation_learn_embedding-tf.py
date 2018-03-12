from __future__ import print_function
import tensorflow as tf
import numpy as np
import re, random, sys
from datetime import datetime
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
token_to_ordinal = dict((token, i) for i, token in enumerate(token_set))
ordinal_to_token = dict((v,k) for k,v in token_to_ordinal.items())
# Write mapping to disk as TSV for tensorboard projector
with open('tokens.tsv', 'w') as f:
  #f.write('token\n')
  for token, ordinal in token_to_ordinal.items():
    f.write(token+'\n')

print("total number of unique tokens", len(token_set))

num_embedding_dims = 4
batch_size = 10
sequence_len = 4
num_embedding_dims = 32
batch_size = 256
sequence_len = 16

features = tf.placeholder(tf.int32, [batch_size, sequence_len], name='features')
labels = tf.placeholder(tf.int32, [batch_size], name='labels')
embedding = tf.Variable(tf.random_normal([len(token_set), num_embedding_dims], stddev=1.0, dtype=tf.float32), name='embedding')
xe = tf.nn.embedding_lookup(embedding, features, name='x_embedded')
ye = tf.nn.embedding_lookup(embedding, labels, name='y_embedded')
xr = tf.reshape(xe, [batch_size, sequence_len * num_embedding_dims])
w = tf.Variable(tf.random_normal([sequence_len * num_embedding_dims, num_embedding_dims], stddev=0.1, dtype=tf.float32), name='weights')
b = tf.Variable(tf.random_normal([num_embedding_dims], stddev=0.1, dtype=tf.float32), name='bias')
y = tf.tanh(tf.matmul(xr, w) + b)
# decode embedding back to ordinal value
decoded_y = tf.argmin(tf.reduce_sum(
  (tf.reshape(y, [batch_size, 1, num_embedding_dims]) -
  tf.reshape(embedding, [1, len(token_set), num_embedding_dims])) ** 2
  , axis=2
  )
, axis=1
, output_type=tf.int32
, name='decoded_y'
)
decoded_x = tf.argmin(tf.reduce_sum(
  (tf.reshape(xe, [batch_size, 1, sequence_len, num_embedding_dims]) -
  tf.reshape(embedding, [1, len(token_set), 1, num_embedding_dims])) ** 2
  , axis=2)
, axis=1
, output_type=tf.int32
, name='decoded_x'
)
foo = (ye - y) ** 2
embedding_mean, embedding_variance = tf.nn.moments(embedding, axes=[0])
main_cost = tf.reduce_mean(foo)
cost = main_cost \
  + tf.reduce_mean((embedding_mean - 0.0) ** 2) \
  + tf.reduce_mean((embedding_variance - 1.0) ** 2)
#optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000001).minimize(cost)

features_data = np.zeros([batch_size, sequence_len], int)
labels_data = np.zeros([batch_size], int)

_logDir = None
def getLogDir():
  global _logDir
  if _logDir is None:
    _logDir = 'log/' + datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
  return _logDir

tf.summary.scalar('cost', cost)
summary_op = tf.summary.merge_all()

with tf.Session() as sess:
  # writer for summaries
  writer = tf.summary.FileWriter(getLogDir(), graph=sess.graph)
  # checkpoint writer for all trainable model variables
  saver = tf.train.Saver()
  # initialize tf variables
  sess.run(tf.global_variables_initializer())
  keep_training = True
  steps = 0
  seq_starts = [0] * batch_size
  while keep_training:
    # generate a sequence to train on
    for iBatchSample in range(batch_size):
      seq_start = random.randint(0, len(corpus_tokens) - sequence_len - 1)
      #seq_start = random.randint(0, (len(corpus_tokens) // (sequence_len + 1))-1) * (sequence_len + 1)
      seq_starts[iBatchSample] = seq_start
      #seq_start = 0
      features_data[iBatchSample] = list(token_to_ordinal[token] for token in corpus_tokens[seq_start:seq_start+sequence_len])
      labels_data[iBatchSample] = token_to_ordinal[corpus_tokens[seq_start+sequence_len]]

    ign, cost_run, main_cost_run, summary = sess.run([optimizer, cost, main_cost, summary_op], feed_dict={
      features: features_data
    , labels: labels_data
    })
    writer.add_summary(summary, steps)

    if steps % 2000 == 0:
      seq_start = random.randint(0, len(corpus_tokens) - sequence_len - 1)
      seed_tokens = corpus_tokens[seq_start:seq_start+sequence_len]
      print('seed:', ' '.join(seed_tokens))
      print('sentence:')
      for i in range(50):
        features_data[0] = list(token_to_ordinal[token] for token in seed_tokens)
        (decoded_y_run,) = sess.run(
          [decoded_y], feed_dict={
          features: features_data
        })
        sys.stdout.write(ordinal_to_token[decoded_y_run[0]] + ' ')
        for i in range(1, len(seed_tokens)):
          seed_tokens[i-1] = seed_tokens[i]
        seed_tokens[-1] = ordinal_to_token[decoded_y_run[0]]
      print()
      print('cost=', cost_run)
      print('main_cost=', main_cost_run)

    if steps % 1000 == 0:
      print('saving')
      saver.save(sess, 'checkpoints/latest.ckpt', steps)
    steps += 1

