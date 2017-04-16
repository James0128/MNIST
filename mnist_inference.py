# -*- coding: utf-8 -*- 2
import tensorflow as tf

#定义神经网络结构相关参数
INPUT_NODE= 784
OUTPUT_NODE =10
LAYER1_NODE=500
#测试中通过保存的模型来加载这些变量取值
def get_weight_variable(shape, regularizer):
    weights=tf.get_variable("weights",shape,initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights
#定义神经网络的前向传播过程
def inference(input_tensor,regularizer):
    #声明第一层神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer1'):
        weights =get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases =tf.get_variable("biases",[LAYER1_NODE],initializer=tf.constant_initializer(0.0))
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
    #类似的声明第二层神经网络的变量并完成前向传播的过程
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases=tf.get_variable("biases",[OUTPUT_NODE],initializer=tf.constant_initializer(0.0))
        layer2=tf.matmul(layer1,weights)+biases
    #返回最后前向传播结果
    return layer2
