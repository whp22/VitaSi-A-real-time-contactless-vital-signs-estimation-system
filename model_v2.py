import tensorflow as tf
import numpy as np
#from utils import *
import os
import h5py
from resnet.utils import *
FPS = 48
LRATE = 0.0000001
W = 96
H = 64
BATCH_SIZE = 32
EPOCHS = 50

train_path = '/home/hwang266/Desktop/Wang/COHFACE/train'
test_path = '/home/hwang266/Desktop/Wang/COHFACE/test'

data = '/home/hwang266/Desktop/Wang/COHFACE/'


def find_file(filepath):
    # Walk through all files in the directory that contains the files to copy
    h5file = []
    for root, dirs, files in os.walk(filepath):
        for filename in files:
            if os.path.splitext(filename)[1] == '.h5':
                h5file.append(os.path.join(root, filename))
    return h5file


def get_data(filepath):
    datalist = []
    labellist = []
    brlist = []
    h5file = find_file(filepath)
    for h5 in h5file:
        with h5py.File(h5, 'r') as f:
            img = f['img']
            hr = f['hr']
            br = f['br']
            datalist.extend(img)
            labellist.extend(hr)
            brlist.extend(br)

    data_array = np.asarray(datalist)
    label_array = np.asarray(labellist)
    label_br = np.asarray(brlist)
    print('data and label shape are', data_array.shape, label_array.shape, label_br.shape)
    return data_array, label_array, label_br


class HR_BR():
    def __init__(self, sess, BR_TRAIN=False):
        self.sess = sess
        self.BR_TRAIN = BR_TRAIN
        if self.BR_TRAIN:
            self.normal = 35
        else:
            self.normal = 240 - 45

        self.input_frames = tf.placeholder(dtype=tf.float32, shape=[None, FPS, FPS, 3])
        self.GT = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        # self.BR=tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.outputs = self.model_v4_mobile(self.input_frames)

        with tf.name_scope('training'):

            # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=c.LRATE, name='optimizer')
            self.optimizer = tf.train.AdamOptimizer(learning_rate=LRATE, name='optimizer')

            if self.BR_TRAIN:

                self.global_loss = tf.reduce_mean(tf.square(self.outputs[1] - self.GT))
                self.rmse = tf.reduce_mean(tf.sqrt(tf.square(self.outputs[1] - self.GT)))
                weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="BR")

                self.train_op = self.optimizer.minimize(self.global_loss, name='train_op', var_list=weights)
            else:
                self.global_loss = tf.reduce_mean(tf.square(self.outputs[0] - self.GT))
                self.rmse = tf.reduce_mean(tf.sqrt(tf.square(self.outputs[0] - self.GT)))
                self.train_op = self.optimizer.minimize(self.global_loss, name='train_op')

        self.sess.run(tf.global_variables_initializer())




    def model_v4_mobile(self, x, is_training=tf.constant(True, dtype=tf.bool), bn_decay=None, reuse=False):
        BR = tf.constant(0)

        with tf.variable_scope('global'):
            net = conv(x, 96, kernel=5, stride=1, padding='VALID', use_bias=True, scope='conv_00')
            net = relu(net)
            net = conv(net, 96, kernel=3, stride=1, padding='VALID', use_bias=True, scope='conv_0')
            net = relu(net)
            net = conv(net, 96, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_02')
            net = relu(net)
            net = conv(net, 96, kernel=3, stride=2, padding='VALID', use_bias=True, scope='conv_1')
            net = relu(net)
            net = conv(net, 96, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_12')
            net = relu(net)
            # net = conv(net, 96, kernel=3, stride=2, padding='VALID', use_bias=True, scope='conv_11')
            # net = relu(net)
            # net = conv(net, 96, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_121')
            # net = relu(net)
            # x = batch_norm(x, is_training, scope='batch_norm2')

            with tf.variable_scope('HR'):
                HR = conv(net, 128, kernel=3, stride=2, padding='VALID', use_bias=True, scope='conv_4')
                # x = batch_norm(x, is_training, scope='batch_norm4')
                HR = relu(HR)
                HR = conv(HR, 128, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_42')
                HR = relu(HR)
                HR = conv(HR, 128, kernel=3, stride=2, padding='VALID', use_bias=True, scope='conv_41')
                # x = batch_norm(x, is_training, scope='batch_norm4')
                HR = relu(HR)
                HR = conv(HR, 128, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_421')
                HR = relu(HR)
                # HR = tf.nn.max_pool(HR, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pooling')
                print(HR)
                HR = tf.nn.avg_pool(HR, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID', name='pooling')

                HR = fully_conneted(HR, units=192, scope='fc1')
                HR = relu(HR)
                # HR = dropout(HR, is_training=is_training, scope='dropout', keep_prob=0.6)
                HR = fully_conneted(HR, units=1, scope='fc2')

        if self.BR_TRAIN:
            with tf.variable_scope('BR'):
                # BR = conv(x, 64, kernel=3, stride=1, padding='VALID', use_bias=True, scope='conv_1')
                # BR = relu(BR)
                BR = conv(net, 128, kernel=3, stride=2, padding='VALID', use_bias=True, scope='conv_4')
                BR = relu(BR)
                BR = conv(BR, 128, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_42')
                BR = relu(BR)
                BR = conv(BR, 256, kernel=3, stride=2, padding='VALID', use_bias=True, scope='conv_5')
                # x = batch_norm(x, is_training, scope='batch_norm4')
                BR = relu(BR)
                BR = conv(BR, 256, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_52')
                BR = relu(BR)
                BR = tf.nn.avg_pool(BR, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID', name='pooling')
                BR = fully_conneted(BR, units=192, scope='fc1')
                BR = relu(BR)
                # BR = dropout(BR, is_training=is_training, scope='dropout', keep_prob=0.6)
                BR = fully_conneted(BR, units=1, scope='fc2')

        outputs = [HR, BR]

        return outputs


def stage_train(train_imgs, train_labels, train_br, test_imgs, test_labels, test_br, stage=1, test_flag=True):
    if stage == 1:
        Flag = False
        model_save = 'tmp/training/HR/model.ckpt'
        EPOCHS = 50

    else:

        Flag = True
        model_save = 'tmp/training/HR_BR/model.ckpt'
        EPOCHS = 10

    with tf.Session() as sess:
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            HB = HR_BR(sess, Flag)
            saver = tf.train.Saver()

            ##load specific weights
            print()
            print()
            if HB.BR_TRAIN:
                weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="global")
                saver_br = tf.train.Saver(var_list=weights)

                saver_br.restore(sess, "tmp/training/HR/model.ckpt")
                print('restore done')
            print("training stage: ", stage)
            for e in range(EPOCHS):

                ids = np.random.permutation(train_imgs.shape[0])
                loss = 0
                count = 0

                for i in range(0, len(ids), BATCH_SIZE):
                    IDX = ids[i:i + BATCH_SIZE]
                    imgs_batch = train_imgs[IDX]
                    label_batch = train_labels[IDX]
                    br_batch = train_br[IDX]

                    if HB.BR_TRAIN:
                        _, outputs, global_loss = HB.sess.run(
                            [HB.train_op, HB.outputs, HB.global_loss],
                            feed_dict={HB.input_frames: imgs_batch, HB.GT: br_batch})
                    else:

                        _, outputs, global_loss = HB.sess.run(
                            [HB.train_op, HB.outputs, HB.global_loss],
                            feed_dict={HB.input_frames: imgs_batch, HB.GT: label_batch})
                    loss += global_loss
                    count = count + 1
                print('epoch： ', e, ' ', loss / count, end=' ')

                # testing
                if test_flag:
                    count = 0
                    loss = 0
                    r = 0
                    for i in range(0, test_imgs.shape[0], BATCH_SIZE):

                        imgs_batch = test_imgs[i:i + BATCH_SIZE]
                        label_batch = test_labels[i:i + BATCH_SIZE]
                        br_batch = test_br[i:i + BATCH_SIZE]

                        if HB.BR_TRAIN:
                            outputs, global_loss, rmse = HB.sess.run(
                                [HB.outputs, HB.global_loss, HB.rmse],
                                feed_dict={HB.input_frames: imgs_batch, HB.GT: br_batch})
                        else:

                            outputs, global_loss, rmse = HB.sess.run(
                                [HB.outputs, HB.global_loss, HB.rmse],
                                feed_dict={HB.input_frames: imgs_batch, HB.GT: label_batch})

                        loss += global_loss
                        r += rmse
                        count = count + 1
                    print('  test_loss： ', loss / count, '  RMSE: ', r * HB.normal / count)

            save_path = saver.save(sess, model_save)


def main():
    train_imgs, train_labels, train_br = get_data(train_path)
    test_imgs, test_labels, test_br = get_data(test_path)
    train_labels = np.expand_dims(train_labels, -1)

    train_br = np.expand_dims(train_br, -1)
    test_br = np.expand_dims(test_br, -1)
    test_labels = np.expand_dims(test_labels, -1)

    print(train_imgs.shape, train_labels.shape)

    #stage_train(train_imgs, train_labels, train_br, test_imgs, test_labels, test_br, stage=1)
    stage_train(train_imgs, train_labels, train_br, test_imgs, test_labels, test_br, stage=2)

    # testing



    with tf.Session() as sess:
        HB = HR_BR(sess,True)
        saver = tf.train.Saver()


        ##load specific weights
        if HB.BR_TRAIN:
            weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="global")
            saver_br=tf.train.Saver(var_list=weights)


            saver_br.restore(sess, "tmp/HR/model.ckpt")


        for e in range(EPOCHS):

            ids = np.random.permutation(train_imgs.shape[0])
            loss = 0
            count = 0

            for i in range(0,len(ids),BATCH_SIZE):
                IDX=ids[i:i+BATCH_SIZE]
                imgs_batch=train_imgs[IDX]
                label_batch=train_labels[IDX]
                br_batch=train_br[IDX]

                with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                    if HB.BR_TRAIN:
                        _, outputs, global_loss = HB.sess.run(
                            [HB.train_op, HB.outputs, HB.global_loss],
                            feed_dict={HB.input_frames: imgs_batch, HB.GT: br_batch})
                    else:

                        _, outputs, global_loss = HB.sess.run(
                            [HB.train_op, HB.outputs, HB.global_loss],
                            feed_dict={HB.input_frames: imgs_batch, HB.GT: label_batch})
                loss+=global_loss
                count=count+1
            print('epoch： ', e, ' ', loss/count)
        save_path = saver.save(sess, "tmp/model.ckpt")




def demo():
    train_imgs, train_labels, br = get_data(train_path)
    test_imgs, test_labels, br = get_data(test_path)
    train_labels = np.expand_dims(train_labels, -1)
    br = np.expand_dims(br, -1)
    test_labels = np.expand_dims(test_labels, -1)
    imgs_placeholder = tf.placeholder(train_imgs.dtype, train_imgs.shape)
    labels_placeholder = tf.placeholder(train_labels.dtype, [train_labels.shape[0], 1])

    dataset = tf.data.Dataset.from_tensor_slices((imgs_placeholder, labels_placeholder))
    dataset = dataset.shuffle(10000).batch(BATCH_SIZE).repeat()
    iterator = dataset.make_initializable_iterator()

    next_element = iterator.get_next()

    with tf.Session() as sess:
        HB = HR_BR(sess, True)

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer,
                 feed_dict={imgs_placeholder: train_imgs, labels_placeholder: train_labels})

        #saver.restore(sess, "tmp/model.ckpt")

        # H = tf.get_default_graph().get_tensor_by_name('global/HR/HR_fc2:0')

        # B = tf.get_default_graph().get_tensor_by_name('BR/BR_fc2')

        for e in range(EPOCHS):
            # for step in range(num_batches):

            imgs_batch, label_batch = sess.run(next_element)



            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                outputs = HB.sess.run(
                    HB.outputs,
                    feed_dict={HB.input_frames: imgs_batch})

                loss1 = np.mean(np.square(outputs[0] - label_batch))
                loss2 = np.mean(np.square(outputs[1] - label_batch))

                print('epoch： ', e, ' ', loss1, " ", loss2)


if __name__ == '__main__':
    #main()
    demo()