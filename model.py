import tensorflow as tf
import numpy as np
from utils import *
import os
import h5py


FPS=48
LRATE=0.001
W=96
H=64
BATCH_SIZE=32
EPOCHS=100

preprocesing = ColoarMagnify(levels=4)
train_path = '/home/hwang266/Desktop/Wang/COHFACE/train'
test_path = '/home/hwang266/Desktop/Wang/COHFACE/test'

data='/home/hwang266/Desktop/Wang/COHFACE/'



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
    brlist=[]
    h5file = find_file(filepath)
    for h5 in h5file:
        with h5py.File(h5, 'r') as f:
            img = f['img']
            hr = f['hr']
            br=f['br']
            datalist.extend(img)
            labellist.extend(hr)
            brlist.extend(br)

    data_array = np.asarray(datalist)
    label_array = np.asarray(labellist)
    label_br=np.asarray(brlist)
    print('data and label shape are', data_array.shape, label_array.shape, label_br.shape)
    return data_array, label_array,label_br

class HR_BR():
    def __init__(self,sess,BR_TRAIN=False):
        self.sess = sess
        self.BR_TRAIN=BR_TRAIN
        if self.BR_TRAIN:
            self.normal=35
        else:
            self.normal=240-45

        self.input_frames = tf.placeholder(dtype=tf.float32, shape=[None, FPS, FPS,3 ])
        self.GT= tf.placeholder(dtype=tf.float32, shape=[None, 1])
        # self.BR=tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.outputs = self.model(self.input_frames)


        with tf.name_scope('training'):


            # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=c.LRATE, name='optimizer')
            self.optimizer = tf.train.AdamOptimizer(learning_rate=LRATE, name='optimizer')

            if self.BR_TRAIN:

                self.global_loss = tf.reduce_mean(tf.square(self.outputs[1] - self.GT))
                self.rmse=tf.reduce_mean(tf.sqrt(tf.square(self.outputs[1] - self.GT)))
                weights=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="BR")

                self.train_op = self.optimizer.minimize(self.global_loss, name='train_op',var_list=weights)
            else:
                self.global_loss = tf.reduce_mean(tf.square(self.outputs[0] - self.GT))
                self.rmse = tf.reduce_mean(tf.sqrt(tf.square(self.outputs[0] - self.GT)))
                self.train_op = self.optimizer.minimize(self.global_loss,name='train_op')


        self.sess.run(tf.global_variables_initializer())



    def model(self, inputs, is_training=tf.constant(True, dtype=tf.bool), bn_decay=None):

            BR=tf.constant(0)
            BN_F=False


            with tf.variable_scope('global'):
                net = conv2d(inputs, 96, [5, 5],
                             padding='VALID', stride=[1, 1],
                             bn=BN_F, is_training=is_training,
                             scope='global_conv1', bn_decay=bn_decay)

                #net=max_pool2d(net,kernel_size=[3,3], stride=[2,2],scope='global_pool1')

                net = conv2d(net, 96, [3, 3],
                             padding='VALID', stride=[1, 1],
                             bn=BN_F, is_training=is_training,
                             scope='global_conv2', bn_decay=bn_decay)

                #net = max_pool2d(net, kernel_size=[3,3], stride=[2,2],scope='global_pool2')
                with tf.variable_scope('HR'):

                    HR = conv2d(net, 128, [3, 3],
                                padding='VALID', stride=[1, 1],
                                bn=BN_F, is_training=is_training,
                                scope='HR_conv3', bn_decay=bn_decay)
                    HR = max_pool2d(HR, kernel_size=[3,3], stride=[2,2],scope='HR_pool')

                    HR=tf.reshape(HR,[-1,128*19*19])

                    HR=fully_connected(HR,192,scope='HR_fc1')

                    HR=dropout(HR,is_training=is_training,scope='HR_dropout',keep_prob=0.6)

                    HR = fully_connected(HR, 1, scope='HR_fc2')
            if self.BR_TRAIN:
                with tf.variable_scope('BR'):

                    BR = conv2d(net, 128, [3, 3],
                                     padding='VALID', stride=[1, 1],
                                     bn=BN_F, is_training=is_training,
                                     scope='BR_conv3', bn_decay=bn_decay)

                    BR = max_pool2d(BR, kernel_size=[3, 3], stride=[2, 2], scope='BR_pool')

                    BR=tf.reshape(BR,[-1,128*19*19])

                    BR = fully_connected(BR, 192, scope='BR_fc1')
                    BR = dropout(BR, is_training=is_training, scope='BR_dropout', keep_prob=0.6)
                    BR = fully_connected(BR, 1, scope='BR_fc2')

            outputs=[HR,BR]

            return outputs

    # def pre(self, face_input):
    #     filtered_img = np.float32(preprocesing.get_filtered_img(face_input))
    #
    #     minVal = np.min(filtered_img)
    #     maxVal = np.max(filtered_img)
    #
    #     scaling_factor = 255.0 / (maxVal - minVal)
    #     add_factor = -minVal * 255.0 / (maxVal - minVal)
    #
    #     filtered_img = np.float32(filtered_img * scaling_factor + add_factor)
    #
    #     #filtered_img = np.transpose(filtered_img, (2, 0, 1))
    #     #filtered_img = np.expand_dims(filtered_img, 0)
    #     filtered_img = filtered_img * 0.0078125 + (-127.5 * 0.0078125)
    #
    #     return filtered_img





def stage_train(train_imgs,train_labels,train_br,test_imgs,test_labels,test_br,stage=1,test_flag=True):

    if stage==1:
        Flag=False
        model_save='tmp/training/HR/model.ckpt'

    else:

        Flag=True
        model_save='tmp/training/HR_BR/model.ckpt'

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

                #testing
                if test_flag:
                    count=0
                    loss=0
                    r=0
                    for i in range(0, test_imgs.shape[0], BATCH_SIZE):

                        imgs_batch = test_imgs[i:i + BATCH_SIZE]
                        label_batch = test_labels[i:i + BATCH_SIZE]
                        br_batch = test_br[i:i + BATCH_SIZE]

                        if HB.BR_TRAIN:
                                outputs, global_loss,rmse = HB.sess.run(
                                    [HB.outputs, HB.global_loss,HB.rmse],
                                    feed_dict={HB.input_frames: imgs_batch, HB.GT: br_batch})
                        else:

                                outputs, global_loss, rmse = HB.sess.run(
                                    [HB.outputs, HB.global_loss, HB.rmse],
                                    feed_dict={HB.input_frames: imgs_batch, HB.GT: label_batch})

                        loss += global_loss
                        r+=rmse
                        count = count + 1
                    print('  test_loss： ', loss / count, '  RMSE: ',r*HB.normal/count)

            save_path = saver.save(sess, model_save)




def main():
    train_imgs, train_labels,train_br = get_data(train_path)
    test_imgs, test_labels,test_br = get_data(test_path)
    train_labels = np.expand_dims(train_labels, -1)

    train_br = np.expand_dims(train_br, -1)
    test_br = np.expand_dims(test_br, -1)
    test_labels = np.expand_dims(test_labels, -1)





    print(train_imgs.shape,train_labels.shape)

    stage_train(train_imgs,train_labels,train_br,test_imgs,test_labels,test_br,stage=1)
    stage_train(train_imgs, train_labels, train_br, test_imgs, test_labels, test_br, stage=2)

    #testing





    '''
    
    
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
        
        #stage 2
        '''






    #imgs_placeholder = tf.placeholder(train_imgs.dtype, train_imgs.shape)
    #labels_placeholder = tf.placeholder(train_labels.dtype, [train_labels.shape[0],1])

    #dataset = tf.data.Dataset.from_tensor_slices((imgs_placeholder, labels_placeholder))
    #dataset = dataset.shuffle(10000).batch(BATCH_SIZE).repeat()
    #iterator = dataset.make_initializable_iterator()
    #next_element = iterator.get_next()

    #with tf.Session() as sess:
    #    HB = HR_BR(sess,False)
    #    saver = tf.train.Saver()
    #
    #    sess.run(tf.global_variables_initializer())
    #    sess.run(iterator.initializer,
    #             feed_dict={imgs_placeholder: train_imgs, labels_placeholder: train_labels})
    #    ##load specific weights
    #    if HB.BR_TRAIN:
    #        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="global")
    #        saver_br=tf.train.Saver(var_list=weights)


    #        saver_br.restore(sess, "tmp/model.ckpt")

    #    for e in range(EPOCHS):
    #        # for step in range(num_batches):

    #        imgs_batch, label_batch = sess.run(next_element)
    #        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):

    #            _, outputs, global_loss = HB.sess.run(
    #                [HB.train_op, HB.outputs, HB.global_loss],
    #                feed_dict={HB.input_frames: imgs_batch, HB.GT: label_batch})
    #
    #            print('epoch： ',e,' ', global_loss)
    #    save_path = saver.save(sess, "tmp/model.ckpt")


def demo():


    train_imgs, train_labels,br = get_data(train_path)
    test_imgs, test_labels,br = get_data(test_path)
    train_labels = np.expand_dims(train_labels, -1)
    br=np.expand_dims(br,-1)
    test_labels = np.expand_dims(test_labels, -1)
    imgs_placeholder = tf.placeholder(train_imgs.dtype, train_imgs.shape)
    labels_placeholder = tf.placeholder(train_labels.dtype, [train_labels.shape[0], 1])

    dataset = tf.data.Dataset.from_tensor_slices((imgs_placeholder, labels_placeholder))
    dataset = dataset.shuffle(10000).batch(BATCH_SIZE).repeat()
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        HB = HR_BR(sess,True)

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer,
                 feed_dict={imgs_placeholder: train_imgs, labels_placeholder: train_labels})

        saver.restore(sess, "tmp/model.ckpt")

        #H = tf.get_default_graph().get_tensor_by_name('global/HR/HR_fc2:0')

        #B = tf.get_default_graph().get_tensor_by_name('BR/BR_fc2')

        for e in range(EPOCHS):
            # for step in range(num_batches):

            imgs_batch, label_batch = sess.run(next_element)

            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                outputs= HB.sess.run(
                    HB.outputs,
                    feed_dict={HB.input_frames: imgs_batch})

                loss1=np.mean(np.square(outputs[0]-label_batch))
                loss2=np.mean(np.square(outputs[1]-label_batch))


                print('epoch： ', e, ' ', loss1," ",loss2)



if __name__ == '__main__':
    main()
    #demo()