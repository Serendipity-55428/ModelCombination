#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: model_cl
@time: 2019/9/22 下午10:53
'''
import tensorflow as tf
import numpy as np
import pandas as pd
from classifier_dataset import LoadFile

def input(dataset, batch_size):
    '''
    按照指定批次大小随机输出训练集中一个批次的特征/标签矩阵
    :param dataset: 数据集特征矩阵(特征经过01编码后的)
    :param batch_size: 批次大小
    :return: 特征矩阵/标签
    '''
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i+batch_size, :]

def spliting(dataset, size):
    '''
    留一法划分训练和测试集
    :param dataset: 特征数据集/标签
    :param size: 测试集大小
    :return: 训练集和测试集特征矩阵/标签
    '''
    #随机得到size大小的交叉验证集
    test_row = np.random.randint(low=0, high=len(dataset)-1, size=(size))
    #从dataset中排除交叉验证集
    train_row = list(filter(lambda x: x not in test_row, range(len(dataset)-1)))
    return dataset[train_row, :], dataset[test_row, :]

def onehot(dataset, class_number):
    '''
    将所有标签按数值大小编码为one-hot稀疏向量
    :param dataset: 数据集特征矩阵,最后一列为半径标签
    :param class_number: onehot编码长度
    :return: 标签半径被编码后的数据集
    '''
    dataset_pd = pd.DataFrame(data=dataset, columns=[str(i) for i in range(dataset.shape[-1]-1)]+['cl'])
    dataset_onehot = np.array([0])
    for cl_number in range(class_number):
        dataset_sub_pd = dataset_pd.loc[dataset_pd['cl'] == cl_number]
        if dataset_sub_pd.values.shape[0] == 0: continue
        one_hot_label = np.zeros(shape=[dataset_sub_pd.values.shape[0], class_number], dtype=np.float32)
        one_hot_label[:, cl_number] = 1
        one_hot_dataset = np.hstack((dataset_sub_pd.values[:, :-1], one_hot_label))
        dataset_onehot = one_hot_dataset if dataset_onehot.any() == 0 else \
            np.vstack((dataset_onehot, one_hot_dataset))
    return dataset_onehot

class ModelCl:
    ''''''
    @staticmethod
    def concat(inputs, axis):
        fun = tf.keras.layers.Lambda(function=tf.concat)
        return fun(inputs=inputs, axis=axis)
    @staticmethod
    def Inception_resnet_v1(input, last_filter):
        '''
        带有残差结构的inception_v1结构
        :param input: 输入特征矩阵
        :param last_filter: 结构中最后一层输出（同输入特征矩阵的feature map尺寸）
        :return: 结构输出
        '''
        layer_line1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[1, 1], padding='same', activation=tf.nn.relu,
                                             kernel_initializer=tf.keras.initializers.TruncatedNormal())(input)
        layer_line1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                             kernel_initializer=tf.keras.initializers.TruncatedNormal())(layer_line1)
        layer_line1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                             kernel_initializer=tf.keras.initializers.TruncatedNormal())(layer_line1)

        layer_line2 = tf.keras.layers.Conv2D(filters=32, kernel_size=[1, 1], padding='same', activation=tf.nn.relu,
                                             kernel_initializer=tf.keras.initializers.TruncatedNormal())(input)
        layer_line2 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                             kernel_initializer=tf.keras.initializers.TruncatedNormal())(layer_line2)

        layer_line3 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                             kernel_initializer=tf.keras.initializers.TruncatedNormal())(input)

        layer_concat = tf.keras.layers.Concatenate(axis=3)(inputs=[layer_line1, layer_line2, layer_line3])
        layer_concat = tf.keras.layers.Conv2D(filters=last_filter, kernel_size=[1, 1], padding='same', activation=tf.nn.relu,
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal())(layer_concat)
        layer_concat = tf.keras.layers.Conv2D(filters=input.get_shape().as_list()[-1], kernel_size=[1, 1], padding='same', activation=tf.nn.relu,
                                           kernel_initializer=tf.keras.initializers.TruncatedNormal())(layer_concat)
        return tf.keras.layers.Add()([input, layer_concat])
    def __init__(self, *args):
        ''''''
        self.__dataset = args[0]
        self.__savepath = args[-1]
    def _net1_pre(self, input):
        ''''''
        layer = tf.keras.layers.LSTM(units=128, dropout=0.8, return_sequences=True, name='lstm1')(input)
        layer = tf.keras.layers.LSTM(units=128, dropout=0.8, return_sequences=False, name='lstm2')(layer)
        layer = tf.keras.layers.Flatten(name='net1_flatten')(layer)
        return layer
    def _densenet(self, layer_pre, *units):
        ''''''
        stack_layers = [layer_pre]
        for i in range(len(units)):
            if len(stack_layers) > 1:
                layers_concat = tf.keras.layers.Concatenate(axis=1)(inputs=stack_layers)
            else:
                layers_concat = layer_pre
            layer = tf.keras.layers.Dense(units=units[i], activation=tf.nn.relu if i != len(units)-1 else tf.nn.softmax,
                                          use_bias=True,
                                          kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                          bias_initializer=tf.keras.initializers.TruncatedNormal())(layers_concat)
            stack_layers.append(layer)
        return stack_layers[-1]
    def _net2_pre(self, input):
        ''''''
        layer = ModelCl.Inception_resnet_v1(input=input, last_filter=64)
        layer = ModelCl.Inception_resnet_v1(input=layer, last_filter=128)
        layer = ModelCl.Inception_resnet_v1(input=layer, last_filter=256)
        layer = ModelCl.Inception_resnet_v1(input=layer, last_filter=256)
        return layer
    def _model_finally(self):
        ''''''
        with tf.name_scope('input'):
            input1 = tf.keras.Input(shape=(4,), name='feature4')
            input2 = tf.keras.Input(shape=(20,), name='avgdens')
            input3 = tf.keras.Input(shape=(100,), name='avgdensFFT')
        with tf.name_scope('cl1'):
            input2_reshape = tf.keras.layers.Reshape(target_shape=[4, 5])(input2)
            layer_lstm = self._net1_pre(input=input2_reshape)
            output1 = tf.keras.layers.Concatenate(axis=1)(inputs=[input1, layer_lstm])
            output1 = self._densenet(output1, 100, 100, 100, 250)
        with tf.name_scope('cl2'):
            input3_reshape = tf.keras.layers.Reshape(target_shape=[10, 10, 1])(input3)
            layer_resnet = self._net2_pre(input=input3_reshape)
            layer_resnet_reshape = tf.keras.layers.Flatten()(layer_resnet)
            output2 = tf.keras.layers.Concatenate(axis=1)(inputs=[input1, layer_lstm, layer_resnet_reshape])
            output2 = self._densenet(output2, 100, 100, 100, 250)
        model = tf.keras.Model(inputs=[input1, input2, input3], outputs=[output1, output2])
        return model
    def graph(self):
        ''''''
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/model_cl')
        model = self._model_finally()
        optimizer = tf.keras.optimizers.SGD(lr=1e-2)
        model.compile(optimizer=optimizer, loss=['categorical_crossentropy']*2, loss_weights=[0.3, 0.7],
                      metrics=['accuracy'])
        print(model.metrics_names)
        model.fit(callbacks=[tensorboard]) #可视化, 参数必须要输入完整
        # print(model.metrics_names)
        train_data, test_data = spliting(self.__dataset, 6000)
        flag = 0
        for epoch in range(20000):
            for train_data_batch in input(dataset=train_data, batch_size=500):
                loss_train, _, _, _, _ = model.train_on_batch(x=[train_data_batch[:, :4], train_data_batch[:, 4:-25]],
                                                          y=train_data_batch[:, -25:])
                if epoch % 100 == 0 and flag == 0:
                    print('第%s轮后训练集损失函数值为: %s' % (epoch, loss_train))
                    flag = 1
            if epoch % 100 == 0:
                _, _, _, _, acc2 = model.evaluate(x=[test_data[:, :4], test_data[:, 4:-25]], y=test_data[:, -25:], verbose=0)
                print('测试集准确率为: %s' % acc2)
            flag = 0
        # model.save(filepath=self.__savepath)

if __name__ == '__main__':
    path = '/home/xiaosong/桌面/pny_cl125.pickle'
    save_path = '/home/xiaosong/桌面/pny_re_cl/graph_cl.h5'
    dataset = LoadFile(path)
    # print(dataset.shape)
    model_cl = ModelCl(dataset, save_path)
    model_cl.graph()






