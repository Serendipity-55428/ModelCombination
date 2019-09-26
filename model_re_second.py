#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@contact: 1243049371@qq.com
@software: Pycharm
@file: model_re_second
@time: 2019/9/26 下午4:23
'''
import tensorflow as tf
from model_cl import ModelCl, spliting, input, LoadFile
from model_re_pre import acc_regression
import numpy as np

class Model_re_second(ModelCl):
    def __init__(self, *args):
        super(Model_re_second).__init__(self, args)
    def _model_finally(self):
        ''''''
        with tf.name_scope('input'):
            input1 = tf.keras.Input(shape=(4,), name='feature4')
            input2 = tf.keras.Input(shape=(20,), name='avgdens')
        with tf.name_scope('regression'):
            input2_reshape = tf.keras.layers.Reshape(target_shape=[4, 5])(input2)
            layer_lstm = self._net1_pre(input=input2_reshape)
            output = tf.keras.layers.Concatenate(axis=1)(inputs=[input1, layer_lstm])
            output = self._densenet(output, 100, 100, 100, 250)
        model = tf.keras.Model(inputs=[input1, input2], outputs=output)
        return model
    def graph(self):
        ''''''
        tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/model_cl')
        model = self._model_finally()
        optimizer = tf.keras.optimizers.SGD(lr=1e-2)
        model.compile(optimizer=optimizer, loss=tf.keras.losses.mean_squared_error, metrics=['accuracy'])
        print(model.metrics_names)
        # model.fit(callbacks=[tensorboard])  # 可视化, 参数必须要输入完整
        # print(model.metrics_names)
        train_data, test_data = spliting(self.__dataset, 6000)
        flag = 0
        for epoch in range(20000):
            for train_data_batch in input(dataset=train_data, batch_size=500):
                loss_train, _ = model.train_on_batch(x=[train_data_batch[:, :4], train_data_batch[:, 4:-25]],
                                                              y=train_data_batch[:, -25:])
                if epoch % 100 == 0 and flag == 0:
                    print('第%s轮后训练集损失函数值为: %s' % (epoch, loss_train))
                    flag = 1
            if epoch % 100 == 0:
                r_res_pred = model.predict(x=[test_data[:, :4], test_data[:, 4:-25]], verbose=0)
                acc1 = acc_regression(Threshold=0.5, y_true=test_data[:, -1][:, np.newaxis], y_pred=r_res_pred)
                if acc1 > 0.96: break
                print('测试集中T=%s acc=%s' % (0.5, acc1))
            flag = 0
        # model.save(filepath=self.__savepath)

if __name__ == '__main__':
    path = '/home/xiaosong/桌面/pny_cl125.pickle'
    save_path = '/home/xiaosong/桌面/pny_re_cl/graph_cl.h5'
    dataset = LoadFile(path)
    # print(dataset.shape)
    model_cl = ModelCl(dataset, save_path)
    model_cl.graph()