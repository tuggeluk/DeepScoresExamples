"""Convolutional Neural Network Estimator for DeepScores Classification, built with Tensorflow
"""

import argparse
import sys

import tensorflow as tf
import Classification_BatchDataset
import TensorflowUtils as utils


FLAGS = None


def deepscores_cnn(image, nr_class):

    # placeholder for dropout input
    keep_prob = tf.placeholder(tf.float32)

    # five layers of 3x3 convolutions, followed by relu, 2x2-maxpool and dropout
    W1 = utils.weight_variable([3, 3, 1, 32], name="W1")
    b1 = utils.bias_variable([32], name="b1")
    conv1 = utils.conv2d_basic(image, W1, b1, name="conv1")
    relu1 = tf.nn.relu(conv1, name="relu1")
    pool1 = utils.max_pool_2x2(relu1)
    dropout1 = tf.nn.dropout(pool1, keep_prob=keep_prob)

    W2 = utils.weight_variable([3, 3, 32, 64], name="W2")
    b2 = utils.bias_variable([64], name="b2")
    conv2 = utils.conv2d_basic(dropout1, W2, b2, name="conv2")
    relu2 = tf.nn.relu(conv2, name="relu2")
    pool2 = utils.max_pool_2x2(relu2)
    dropout2 = tf.nn.dropout(pool2, keep_prob=keep_prob)

    W3 = utils.weight_variable([3, 3, 64, 128], name="W3")
    b3 = utils.bias_variable([128], name="b3")
    conv3 = utils.conv2d_basic(dropout2, W3, b3, name="conv3")
    relu3 = tf.nn.relu(conv3, name="relu3")
    pool3 = utils.max_pool_2x2(relu3)
    dropout3 = tf.nn.dropout(pool3, keep_prob=keep_prob)

    W4 = utils.weight_variable([3, 3, 128, 256], name="W4")
    b4 = utils.bias_variable([256], name="b4")
    conv4 = utils.conv2d_basic(dropout3, W4, b4, name="conv4")
    relu4 = tf.nn.relu(conv4, name="relu4")
    pool4 = utils.max_pool_2x2(relu4)
    dropout4 = tf.nn.dropout(pool4, keep_prob=keep_prob)


    W5 = utils.weight_variable([3, 3, 256, 512], name="W5")
    b5 = utils.bias_variable([512], name="b5")
    conv5 = utils.conv2d_basic(dropout4, W5, b5, name="conv5")
    relu5 = tf.nn.relu(conv5, name="relu5")
    pool5 = utils.max_pool_2x2(relu5)
    dropout5 = tf.nn.dropout(pool5, keep_prob=keep_prob)

    # two fully connected layers
    # downsampled 5 times so feature maps should be 32 times smaller
    # size is 7*4*512
    W_fc1 = utils.weight_variable([7*4*512, 1024])
    b_fc1 = utils.bias_variable([1024])

    dropout5_flat = tf.reshape(dropout5, [-1, 7*4*512])
    h_fc1 = tf.nn.relu(tf.matmul(dropout5_flat, W_fc1) + b_fc1)

    W_fc2 = utils.weight_variable([1024, nr_class])
    b_fc2 = utils.bias_variable([nr_class])

    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    return y_conv, keep_prob


def main(unused_argv):
    print("Setting up image reader...")
    data_reader = Classification_BatchDataset.class_dataset_reader(FLAGS.data_dir)
    data_reader.read_images()

    # input-data placeholder
    x = tf.placeholder(tf.float32, [None, data_reader.tile_size[0],data_reader.tile_size[1],1])

    # input-label placeholder
    y_ = tf.placeholder(tf.float32, [None, data_reader.nr_classes])

    # Build the graph for the deep net
    y_conv, keep_prob = deepscores_cnn(x, data_reader.nr_classes)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(5):
            batch = data_reader.next_batch(FLAGS.batch_size)
            if i % 1000 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            if i % 100 == 0:
                _, cross_ent = sess.run([train_step, cross_entropy],feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.8})
                print(cross_ent)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.8})

        # import PIL
        # import pandas as pa
        # batch_nr = 2
        # PIL.Image.fromarray(np.squeeze(batch[0][batch_nr], -1)).show()
        #
        # class_names = pa.read_csv("../Datasets/DeepScores/classification_data" + "/class_names.csv", header=None)
        # print(class_names[1][np.where(batch[1][batch_nr] == 1)[0][0]])

        test_images, test_labels = data_reader.get_test_records()
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: test_images[0:FLAGS.test_batch_size], y_: test_labels[0:FLAGS.test_batch_size], keep_prob: 1.0}))

        # Save the variables to disk.
        save_path = saver.save(sess, FLAGS.model_path)
        print("Model saved in file: %s" % save_path)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/Users/tugg/Documents/DeepScores_datasets_old/DeepScores2017_classification',
                      help='Directory for storing input data')
  parser.add_argument("--batch_size", type=int, default=2, help="batch size for training")
  parser.add_argument("--test_batch_size", type=int, default=200, help="batch size for training")
  parser.add_argument("--model_path", type=str, default="./Models/deepscores_class.ckpt",
                      help="where to store the trained model")

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
