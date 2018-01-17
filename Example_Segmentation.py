""" Fully Convolutional Neural Network Estimator for DeepScores Segmentation, built with Tensorflow
"""
import argparse
import sys
import os
import tensorflow as tf
import Segmentation_BatchDataset
import TensorflowUtils as utils
import datetime


FLAGS = None


def conv_layer(input,r_field,input_c,out_c,nr):
    W = utils.weight_variable([r_field, r_field, input_c, out_c], name="W"+str(nr))
    b = utils.bias_variable([out_c], name="b"+str(nr))
    conv = utils.conv2d_basic(input, W, b, name="conv"+str(nr))
    relu = tf.nn.relu(conv, name="relu"+str(nr))
    return relu


def deconv_layer(input,r_field,in_channels,out_channels, out_shape,nr, stride=2):
    W = utils.weight_variable([r_field, r_field, out_channels, in_channels], name="W_t"+nr)
    b = utils.bias_variable([out_channels], name="b_t"+nr)
    conv_t1 = utils.conv2d_transpose_strided(input, W, b, out_shape)
    return conv_t1


def segment(image, keep_prob_conv, input_channels, output_channels, scope):

    with tf.variable_scope(scope):

        ###############
        # downsample  #
        ###############

        W2 = utils.weight_variable([3, 3, input_channels, 64], name="W2")
        b2 = utils.bias_variable([64], name="b2")
        conv2 = utils.conv2d_basic(image, W2, b2, name="conv2")
        relu2 = tf.nn.relu(conv2, name="relu2")
        pool2 = utils.max_pool_2x2(relu2)
        dropout2 = tf.nn.dropout(pool2, keep_prob=keep_prob_conv)

        W3 = utils.weight_variable([3, 3, 64, 128], name="W3")
        b3 = utils.bias_variable([128], name="b3")
        conv3 = utils.conv2d_basic(dropout2, W3, b3, name="conv3")
        relu3 = tf.nn.relu(conv3, name="relu3")
        pool3 = utils.max_pool_2x2(relu3)
        dropout3 = tf.nn.dropout(pool3, keep_prob=keep_prob_conv)

        W4 = utils.weight_variable([3, 3, 128, 256], name="W4")
        b4 = utils.bias_variable([256], name="b4")
        conv4 = utils.conv2d_basic(dropout3, W4, b4, name="conv4")
        relu4 = tf.nn.relu(conv4, name="relu4")
        pool4 = utils.max_pool_2x2(relu4)
        dropout4 = tf.nn.dropout(pool4, keep_prob=keep_prob_conv)

        W5 = utils.weight_variable([3, 3, 256, 512], name="W5")
        b5 = utils.bias_variable([512], name="b5")
        conv5 = utils.conv2d_basic(dropout4, W5, b5, name="conv5")
        relu5 = tf.nn.relu(conv5, name="relu5")
        pool5 = utils.max_pool_2x2(relu5)
        dropout5 = tf.nn.dropout(pool5, keep_prob=keep_prob_conv)

        W6 = utils.weight_variable([3, 3, 512, 512], name="W6")
        b6 = utils.bias_variable([512], name="b6")
        conv6 = utils.conv2d_basic(dropout5, W6, b6, name="conv6")
        relu6 = tf.nn.relu(conv6, name="relu6")
        pool6 = utils.max_pool_2x2(relu6)
        dropout6 = tf.nn.dropout(pool6, keep_prob=keep_prob_conv)

        W7 = utils.weight_variable([3, 3, 512, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(dropout6, W7, b7, name="conv7")

        ############
        # upsample #
        ############

        deconv_shape1 = pool5.get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, 4096], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv7, W_t1, b_t1, output_shape=tf.shape(pool5))

        stacked_1 = tf.concat([conv_t1, pool5], -1)
        fuse_1_1 = conv_layer(stacked_1, 1, 2*deconv_shape1[3].value, deconv_shape1[3].value, "fuse_1_1")
        fuse_1_2 = conv_layer(fuse_1_1, 1, deconv_shape1[3].value, deconv_shape1[3].value, "fuse_1_2")

        deconv_shape2 = pool4.get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1_2, W_t2, b_t2, output_shape=tf.shape(pool4))

        stacked_2 = tf.concat([conv_t2, pool4], -1)
        fuse_2_1 = conv_layer(stacked_2, 1, 2*deconv_shape2[3].value, deconv_shape2[3].value, "fuse_2_1")
        fuse_2_2 = conv_layer(fuse_2_1, 1, deconv_shape2[3].value, deconv_shape2[3].value, "fuse_2_2")

        deconv_shape3 = pool3.get_shape()
        W_t3 = utils.weight_variable([4, 4, deconv_shape3[3].value, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([deconv_shape3[3].value], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2_2, W_t3, b_t3, output_shape=tf.shape(pool3))

        stacked_3 = tf.concat([conv_t3, pool3], -1)
        fuse_3_1 = conv_layer(stacked_3, 1, 2*deconv_shape3[3].value, deconv_shape3[3].value, "fuse_3_1")
        fuse_3_2 = conv_layer(fuse_3_1, 1, deconv_shape3[3].value, deconv_shape3[3].value, "fuse_3_2")

        deconv_shape4 = pool2.get_shape()
        W_t4 = utils.weight_variable([4, 4, deconv_shape4[3].value, deconv_shape3[3].value], name="W_t4")
        b_t4 = utils.bias_variable([deconv_shape4[3].value], name="b_t4")
        conv_t4 = utils.conv2d_transpose_strided(fuse_3_2, W_t4, b_t4, output_shape=tf.shape(pool2))

        stacked_4 = tf.concat([conv_t4, pool2], -1)
        fuse_4_1 = conv_layer(stacked_4, 1, 2*deconv_shape4[3].value, deconv_shape4[3].value, "fuse_4_1")
        fuse_4_2 = conv_layer(fuse_4_1, 1, deconv_shape4[3].value, deconv_shape4[3].value, "fuse_4_2")

        # do the final upscaling
        shape = tf.shape(image)
        deconv_shape5 = tf.stack([shape[0], shape[1], shape[2], output_channels])
        W_t5 = utils.weight_variable([16, 16, output_channels, deconv_shape4[3].value], name="W_t5")
        b_t5 = utils.bias_variable([output_channels], name="b_t5")
        conv_t5 = utils.conv2d_transpose_strided(fuse_4_2, W_t5, b_t5, output_shape=deconv_shape5, stride =2)


    annotation_pred = tf.argmax(conv_t5, dimension=3, name="prediction")
    return tf.expand_dims(annotation_pred, dim=3), conv_t5


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(unused_argv):
    print("Setting up image reader...")
    data_reader = Segmentation_BatchDataset.seg_dataset_reader(FLAGS.data_dir, crop=FLAGS.crop, crop_size=FLAGS.crop_size)
    print("Images read")

    #Placeholders for FeedDict
    keep_probability_conv = tf.placeholder(tf.float32, name="keep_probability_conv")
    image = tf.placeholder(tf.float32, shape=[None, FLAGS.crop_size[0], FLAGS.crop_size[0], 1], name="image")
    annotation = tf.placeholder(tf.int32, shape=[None, FLAGS.crop_size[0], FLAGS.crop_size[0], 1], name="labels")

    # Apply FCN
    pred_annotation, logits = segment(image, keep_probability_conv, 1, FLAGS.nr_classes, "labels")

    # compute cross-entropy loss
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="loss_labels")))
    # set up adam-optimizer
    trainable_var = tf.trainable_variables()
    train_op = train(loss, trainable_var)

    # get TF session
    sess = tf.Session()

    # set up saver
    saver = tf.train.Saver()


    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1]) # get the step from the last checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        step = 0

    for itr in xrange(step, FLAGS.MAX_ITERATION):
        train_images, train_annotations= data_reader.next_batch(FLAGS.batch_size)
        feed_dict = {image: train_images, annotation: train_annotations, keep_probability_conv: 0.85}
        sess.run(train_op, feed_dict=feed_dict)

        print(itr)

        if itr % 10 == 0:
            train_loss, summary_str = sess.run([loss], feed_dict=feed_dict)
            print("Step: %d, Train_loss: %g" % (itr, train_loss))

        if itr % 500 == 0 and itr != 0:
            valid_images, valid_annotations, valid_o_annotations = data_reader.get_test_records()
            valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations, keep_probability_conv: 1.0})
            print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
            saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/Users/tugg/datasets/DeepScores',
                      help='Directory for storing input data')
  parser.add_argument("--batch_size", type=int, default=1, help="batch size for training")
  parser.add_argument("--crop", type=bool, default=True, help="batch size for training")
  parser.add_argument("--crop_size", type=bytearray, default=[1000,1000], help="batch size for training")
  parser.add_argument("--nr_classes", type=int, default=124, help="batch size for training")
  parser.add_argument("--logs_dir", type=str, default="logs/", help="path to logs directory")
  parser.add_argument("--MAX_ITERATION", type=int, default=50000, help="path to logs directory")
  parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for Adam Optimizer")
  parser.add_argument("--debug", type=bool, default=False, help="debug yes/no")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)