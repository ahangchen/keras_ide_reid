import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from random import shuffle
# from utils import cuda_util

import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.initializers import RandomNormal
from keras.layers import Dense
from keras.models import Model
from keras.models import load_model
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from baseline.evaluate import market_result_eval, lmp_test_predict, grid_result_eval


def load_gan_data(target, LIST, TRAIN):
    images, labels = [], []
    with open(LIST, 'r') as f:
        last_label = -1
        label_cnt = -1
        for line in f:
            line = line.strip()
            img = line
            lbl = line.split('_')[0]
            if last_label != lbl:
                label_cnt += 1
            last_label = lbl
            img = image.load_img(os.path.join(TRAIN, target + '_' + img), target_size=[224, 224])
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            images.append(img[0])
            labels.append(label_cnt)

    img_cnt = len(labels)
    shuffle_idxes = range(img_cnt)
    shuffle(shuffle_idxes)
    shuffle_imgs = list()
    shuffle_labels = list()
    for idx in shuffle_idxes:
        shuffle_imgs.append(images[idx])
        shuffle_labels.append(labels[idx])
    images = np.array(shuffle_imgs)
    labels = to_categorical(shuffle_labels)
    return images, labels


def softmax_transfer(source_model_path, target, train_list, train_dir, class_count, target_model_path):
    images, labels = load_gan_data(target, train_list, train_dir)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    set_session(sess)

    base_model = load_model(source_model_path)
    x = base_model.layers[-2].output
    x = Dense(class_count, activation='softmax', name='fc8', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(x)
    net = Model(inputs=[base_model.input], outputs=[x])

    for layer in net.layers:
        layer.trainable = True

    batch_size = 16
    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        width_shift_range=0.2,  # 0.
        height_shift_range=0.2)

    net.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    net.fit_generator(
        train_datagen.flow(images, labels, batch_size=batch_size),
        steps_per_epoch=len(images) / batch_size + 1, epochs=40,
    )
    net.save(target_model_path)

    return net


def gan_transfer_on_dataset(source, target, project_path='/home/cwh/coding/rank-reid',
                            dataset_parent='/hdd/cwh/spgan_test_prediction/'):
    if source == 'market':
        train_list = project_path + '/dataset/market_train.list'
        class_count = 751
    elif source == 'grid':
        train_list = project_path + '/dataset/grid_train.list'
        class_count = 250
    elif source == 'cuhk':
        train_list = project_path + '/dataset/cuhk_train.list'
        class_count = 971
    elif source == 'viper':
        train_list = project_path + '/dataset/viper_train.list'
        class_count = 630
    elif source == 'duke':
        train_list = project_path + '/dataset/duke_train.list'
        class_count = 702
    elif 'grid-cv' in source:
        cv_idx = int(source.split('-')[-1])
        train_list = project_path + '/dataset/grid-cv/%d.list' % cv_idx
        class_count = 125
    else:
        train_list = 'unknown'
        class_count = -1
    train_dir = dataset_parent + '%s2%s_spgan/bounding_box_train_%s2%s' % (source, target, source, target)
    source_model_path = '../pretrain/' + source + '_softmax_pretrain.h5'
    target_model_path = '../pretrain/%s2%s_softmax_transfer.h5' % (source , target)
    return softmax_transfer(source_model_path, target, train_list, train_dir, class_count, target_model_path)


def get_test_path(source, target):
    dataset_parent_dir = '/home/cwh/coding/'
    if target == 'market':
        target_dir_name = 'Market-1501'
    elif 'grid-cv' in target:
        cv_idx = int(target.split('-')[-1])
        target_dir_name = 'grid_train_probe_gallery/cross%d/' % cv_idx
    elif target == 'duke':
        target_dir_name = 'DukeMTMC-reID'
    probe_path = dataset_parent_dir + '%s/probe' % target_dir_name
    gallery_path = dataset_parent_dir + '%s/test' % target_dir_name
    pid_path = '../data/%s_%s_transfer_pid.log' % (source,  target)
    score_path = '../data/%s_%s_transfer_score.log' % (source,  target)
    if 'grid-cv' in target:
        probe_path = dataset_parent_dir + 'grid_train_probe_gallery' + target.replace('grid-cv-', '/cross') + '/probe'
        gallery_path = dataset_parent_dir + 'grid_train_probe_gallery' + target.replace('grid-cv-', '/cross') + '/test'
    return probe_path, gallery_path, pid_path, score_path


if __name__ == '__main__':
    # sources = ['market']
    sources = ['cuhk', 'duke', 'viper']
    targets = ['grid-cv-%d' % i for i in range(10)]
    for source in sources:
        net = gan_transfer_on_dataset(source, 'grid')
        for target in targets:
            probe_path, gallery_path, pid_path, score_path = get_test_path(source, target)
            target_model_path = '../pretrain/%s2%s_softmax_transfer.h5' % (source, 'grid')
            net = load_model(target_model_path)
            lmp_test_predict(net, probe_path, gallery_path, pid_path, score_path)
            grid_result_eval(pid_path)
    # 'grid',
    sources = ['viper','cuhk', 'grid', 'market', 'duke']
    targets = ['market', 'duke']
    for source in sources:
        for target in targets:
            if target == source:
                continue
            net = gan_transfer_on_dataset(source, target)
            probe_path, gallery_path, pid_path, score_path = get_test_path(source, target)
            target_model_path = '../pretrain/%s2%s_softmax_transfer.h5' % (source, target)
            # net = load_model(target_model_path)
            lmp_test_predict(net, probe_path, gallery_path, pid_path, score_path)
            market_result_eval(pid_path, TEST=gallery_path, QUERY=probe_path)


