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

class ModelCl:
    ''''''
    @staticmethod
    def concat(inputs):
        return tf.concat(values=inputs, axis=1)
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

        layer_concat = ModelCl.concat([layer_line1, layer_line2, layer_line3])
        layer_concat = tf.keras.layers.Conv2D(filters=last_filter, kernel_size=[1, 1], padding='same', activation=tf.nn.relu,
                                              kernel_initializer=tf.keras.initializers.TruncatedNormal())(layer_concat)
        return ModelCl.concat([layer_concat, input])
    def __init__(self, *args):
        ''''''
        self.__feature4 = args[0]
        self.__avgdens = args[1]
        self.__avgdensFFT = args[2]
    def _net1_pre(self, input1, input2):
        ''''''
        layer = tf.keras.layers.LSTM(units=128, dropout=0.8, return_sequences=True, name='lstm1')(input2)
        layer = tf.keras.layers.LSTM(units=128, dropout=0.8, return_sequences=False, name='lstm2')(layer)
        layer = tf.keras.layers.Flatten(name='net1_flatten')(layer)
        layer = tf.keras.layers.Lambda(ModelCl.concat, name='concat1')(inputs=[input1, layer])
        return layer
    def _densenet(self, layer_pre, *units):
        ''''''
        stack_layers = [layer_pre]
        for i in range(len(units)):
            layers_concat = ModelCl.concat(inputs=stack_layers)
            layer = tf.keras.layers.Dense(units=units[i], activation=tf.nn.relu if i != len(units)-1 else tf.nn.softmax,
                                          use_bias=True,
                                          kernel_initializer=tf.keras.initializers.TruncatedNormal(),
                                          bias_initializer=tf.keras.initializers.TruncatedNormal(),
                                          name='fc%s' % (i+1))(layers_concat)
            stack_layers.append(layer)
        return stack_layers[-1]
    def _net2_pre(self, input1, input2, input3):
        ''''''
        layer = ModelCl.concat([input1, input2, input3])
        layer = ModelCl.Inception_resnet_v1(input=layer, last_filter=64)
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
            layer_lstm = self._net1_pre(input1=input1, input2=input2_reshape)
            output1 = self._densenet(layer_lstm, 100, 100, 100, 250)
        with tf.name_scope('cl2'):
            input3_reshape = tf.keras.layers.Reshape(target_shape=[10, 10, 1])(input3)
            layer_resnet = self._net2_pre(input1=input1, input2=layer_lstm, input3=input3_reshape)
            output2 = self._densenet(layer_resnet, 100, 100, 100, 250)
        model = tf.keras.Model(inputs=[input1, input2, input3], outputs=[output1, output2])
        return model
    def graph(self):
        ''''''
        model = self._model_finally()
        optimizer = tf.keras.optimizers.SGD(lr=1e-2)
        model.compile(optimizer=optimizer, loss=['categorical_crossentropy']*2, metrics=['accuracy'])
        print(model.metrics_names)
        # train_data, test_data = spliting(dataset, 6000)
        # flag = 0
        # for epoch in range(20000):
        #     for train_data_batch in input(dataset=train_data, batch_size=500):
        #         loss_train, _ = model.train_on_batch(x=[train_data_batch[:, :4], train_data_batch[:, 4:-25]],
        #                                                   y=train_data_batch[:, -25:])
        #         if epoch % 100 == 0 and flag == 0:
        #             print('第%s轮后训练集损失函数值为: %s' % (epoch, loss_train))
        #             flag = 1
        #     if epoch % 100 == 0:
        #         _, acc = model.evaluate(x=[test_data[:, :4], test_data[:, 4:-25]], y=test_data[:, -25:], verbose=0)
        #         print('测试集准确率为: %s' % acc)
        #     flag = 0
        # # print(model_cl25.get_config())
        # model_cl25.save(filepath=save_path)

if __name__ == '__main__':
    pass






