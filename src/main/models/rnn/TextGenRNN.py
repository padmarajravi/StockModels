from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


logs_path = '/tmp/tensorflow/rnn_words'
writer = tf.summary.FileWriter(logs_path)

training_file = 'flavour_source.txt'

n_input = 3


def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        content = [word for i in range(len(content)) for word in content[i].split()]
        content = np.array(content)
    return content


def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

def RNN(x,weights,biases):

    # reshaping into rows with 3 columns.
    x = tf.reshape(x,[-1,n_input])
    print(tf.shape(x))
    x = tf.split(x,n_input,1)
    print(tf.shape(x))
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']



if __name__ == '__main__':
    training_data = read_data(training_file)
    print("Training data :"+str(training_data))
    dict,revdict = build_dataset(training_data)
    print(dict)
    learning_rate = 0.001
    training_iters = 10000
    display_step = 1000
    n_input = 3
    vocab_size = len(dict)

# number of units in RNN cell
    n_hidden = 512

# tf Graph input
    x = tf.placeholder("float", [None, n_input, 1])
    y = tf.placeholder("float", [None, vocab_size])

# RNN output node weights and biases
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([vocab_size]))
    }

    pred = RNN(x,weights,biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    # Model evaluation
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        step = 0
        offset = random.randint(0,n_input+1)
        end_offset = n_input+1
        acc_total = 0
        loss_total = 0
        writer.add_graph(session.graph)

        while(step < training_iters):
            if(offset> len(training_data) - end_offset):
                offset = random.randint(0, n_input+1)
           # print("Start:"+str(offset))
            # Words encoded to numbers in batches of n_input
            word_batch_encoded = [dict[training_data[i]] for i in range(offset,offset+n_input)]
            word_batch_encoded = np.reshape(np.array(word_batch_encoded), [-1, n_input, 1])
            #print(word_batch_encoded)
            output_word_one_hot = np.zeros((len(dict)),dtype=float)
            #print("Output_word:"+str(training_data[offset+n_input]))
            output_word_one_hot[dict[str(training_data[offset+n_input])]] = 1.0
            #print(output_word_one_hot)
            output_word_one_hot = np.reshape(output_word_one_hot,[1,-1])
            #print(output_word_one_hot)
            _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                    feed_dict={x: word_batch_encoded, y: output_word_one_hot})


            loss_total += loss
            acc_total += acc
            if (step+1) % display_step == 0:
                 print("Iter= " + str(step+1) + ", Average Loss= " + \
                       "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                       "{:.2f}%".format(100*acc_total/display_step))
                 acc_total = 0
                 loss_total = 0
                 symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
                 symbols_out = training_data[offset + n_input]
                 symbols_out_pred = revdict[int(tf.argmax(onehot_pred, 1).eval())]
                 print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))

            step = step + 1
            offset = offset + n_input + 1

        print("Optimization Finished!")
        print("Elapsed time: ", elapsed(time.time() - start_time))
        print("Run on command line.")
        print("\ttensorboard --logdir=%s" % (logs_path))
        print("Point your web browser to: http://localhost:6006/")



        while True:
            prompt = "%s words: " % n_input
            sentence = input(prompt)
            sentence = sentence.strip()
            words = sentence.split(' ')
            if len(words) != n_input:
                continue
            try:
                symbols_in_keys = [dict[str(words[i])] for i in range(len(words))]
                for i in range(32):
                    keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                    onehot_pred = session.run(pred, feed_dict={x: keys})
                    onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                    sentence = "%s %s" % (sentence,revdict[onehot_pred_index])
                    symbols_in_keys = symbols_in_keys[1:]
                    symbols_in_keys.append(onehot_pred_index)
                print(sentence)
            except:
                print("Word not in dictionary")










