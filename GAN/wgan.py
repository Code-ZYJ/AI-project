import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
#%%  预处理
(train_images,train_labels),(_,_)=tf.keras.datasets.mnist.load_data()
train_images=train_images.reshape(train_images.shape[0],28,28,1).astype('float32')-127.5/127.5   #归一化（-1，1）

BATCH_SIZE=32
BUFFER_SIZE=60000

datasets=tf.data.Dataset.from_tensor_slices(train_images)
datasets=datasets.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


#%%  
def generator_model():  # 生成器
    model=tf.keras.Sequential()
    model.add(layers.Dense(256,input_shape=(100,),use_bias=False)) #输入是100维的随机向量，不使用偏置
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(512,use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
        
    model.add(layers.Dense(28*28*1,use_bias=False,activation='tanh'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Reshape((28,28,1)))    #reshape成图片 注意这里取值范围是（-1，1）
    return model



def discriminator_model():  #辨别器
    model=keras.Sequential()
    model.add(layers.Flatten())    #将输入的（28，28，1）打平
   
    model.add(layers.Dense(512,use_bias=False))    
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(256,use_bias=False))    
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(1))
    
    return model



cross_entropy=tf.keras.losses.BinaryCrossentropy(from_logits=True)   #因为最后的loss输出没有激活，所以要有  from_logits=True

def discriminator_loss(real_out,fake_out):  #损失函数
    real_loss=cross_entropy(tf.ones_like(real_out),real_out)  #tf.ones_like是创建一个唯独与tensor相同，元素全为1的tensor
    fake_loss=cross_entropy(tf.zeros_like(fake_out),fake_out)
    return real_loss+fake_loss
def generator_loss(fake_out):  #损失函数  
    return cross_entropy(tf.ones_like(fake_out),fake_out)
    
generator_opt=tf.keras.optimizers.Adam(1e-4)
discriminator_opt=tf.keras.optimizers.Adam(1e-4) 


#%%  训练过程
EPOCHS=100
noise_dim=100 

num_exp_to_generate=16

seed=tf.random.normal([num_exp_to_generate,noise_dim])   #生成16个长度为100的随机向量

generator=generator_model()        #获取生成器模型
discriminator=discriminator_model()     #获取判别器模型

def train_step(images):
    noise=tf.random.normal([num_exp_to_generate,noise_dim])
    
    with tf.GradientTape() as gen_tape,  tf.GradientTape() as disc_tape:
        real_out=discriminator(images,training=True)
        
        gen_image=generator(noise,training=True)
        fake_out=discriminator(gen_image,training=True)
        
        gen_loss=generator_loss(fake_out)
        disc_loss=discriminator_loss(real_out, fake_out)
    gradient_gen=gen_tape.gradient(gen_loss,generator.trainable_variables)
    gradient_disc=disc_tape.gradient(disc_loss,discriminator.trainable_variables)
    generator_opt.apply_gradients(zip(gradient_gen,generator.trainable_variables))
    discriminator_opt.apply_gradients(zip(gradient_disc,discriminator.trainable_variables))

def generator_plot_image(gen_model,test_noise):
    pre_images=gen_model(test_noise,training=False)
    fig=plt.figure(figsize=(4,4))
    for i in range(pre_images.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow((pre_images[i,:,:,0]+1)/2,cmap='gray')
        plt.axis('off')     #不显示坐标系
    plt.show()

def train(dataset,epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)
            print('.',end='')
        generator_plot_image(generator,seed)

train(datasets,EPOCHS)
generator_plot_image(generator,seed)











