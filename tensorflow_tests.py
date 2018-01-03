import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Load MNIST data from tf builtin datasets
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Placeholders
x_ = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x_, [-1, 28, 28, 1])

# First 5x5 Conv ReLU layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# And a 2x2 max_pool layer reduces the 28x28 image to 14x14
h_pool1 = max_pool_2x2(h_conv1)

# Second 5x5 Conv ReLU layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# Another 2x2 max_pool reduces the image to 7x7
h_pool2 = max_pool_2x2(h_conv2)

# Flatten the 7x7 image into a batch of rank 1 vectors
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

# Another ReLU layer
W_fc1 = weight_variable([7 * 7 * 64, 64])
b_fc1 = bias_variable([64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Final (readout) layer
W_fc2 = weight_variable([64, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Set up the loss functions, optimizer, etc
saver = tf.train.Saver()
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state("./model/")
    if ckpt and ckpt.model_checkpoint_path:
        print("Importing meta graph...")
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta")
        print("Restoring saved model...")
        saver.restore(sess, ckpt.model_checkpoint_path)

    print("\nTesting starting accuracy...")
    starting_accuracy = float(accuracy.eval(feed_dict={x_: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    last_checkpoint_accuracy = starting_accuracy
    print("Starting test accuracy: %g\n" % starting_accuracy)
    start_time = time.time()
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x_: batch[0], y_: batch[1], keep_prob: 1.0})
            print("Step %d, training accuracy %g" % (i, train_accuracy))
        if i % 1000 == 0 and i > 0:
            print("\nPerforming checkpoint operations...")
            current_accuracy = float(
                accuracy.eval(feed_dict={x_: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
            accuracy_delta = current_accuracy - last_checkpoint_accuracy
            print("Last checkpoint's test accuracy:     %g" % last_checkpoint_accuracy)
            print("Current test accuracy:               %g" % current_accuracy)
            last_checkpoint_accuracy = current_accuracy
            print("Test accuracy delta this checkpoint: %g" % accuracy_delta)
            print("Saving current model...")
            saver.save(sess, "./model/mnist-model")
            time_elapsed = time.time() - start_time
            start_time = time.time()
            print("Checkpoint time: %d seconds" % time_elapsed)
            print("Resuming training\n")
        train_step.run(feed_dict={x_: batch[0], y_: batch[1], keep_prob: 0.5})

    print("Saving final model/graph...")
    saver.save(sess, "./model/mnist-model")
    print("Testing ending accuracy...")
    ending_accuracy = float(accuracy.eval(feed_dict={x_: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    print("Ending test accuracy: %g\n" % ending_accuracy)
    accuracy_delta = ending_accuracy - starting_accuracy
    print("Test accuracy delta: %g" % accuracy_delta)
