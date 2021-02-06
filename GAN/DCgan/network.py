import tensorflow as tf

#%%
def discriminator_model():    #判别器
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64,(5,5),padding='same',input_shape=(64,64,3),activation='tanh'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))   #使宽和高各缩小2
    model.add(tf.keras.layers.Conv2D(128,(5,5),padding='same',activation='tanh'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(128,(5,5),padding='same',activation='tanh'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Flatten())   #打平
    model.add(tf.keras.layers.Dense(1024,activation='tanh'))
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
    return model

#定义生成器模型
#从随机数来生成图片
def generator_model():         #生成器
    model=tf.keras.models.Sequential()
    # 输入的维度时100，输出维度（神经元个数）1024
    model.add(tf.keras.layers.Dense(input_dim=100,units=1024,activation='tanh'))
    model.add(tf.keras.layers.Dense(128*8*8))      #8192个神经元的全连接层
    model.add(tf.keras.layers.BatchNormalization())  #批标准化，防止收敛过程变慢，抑制梯度弥散
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.Reshape((8,8,128)))      # 8*8像素的图片
    model.add(tf.keras.layers.UpSampling2D(size=(2,2)))    #16*16像素的图片了
    model.add(tf.keras.layers.Conv2D(128,(5,5),padding='same',activation='tanh'))
    model.add(tf.keras.layers.UpSampling2D(size=(2,2)))    #32*32像素的图像
    model.add(tf.keras.layers.Conv2D(128,(5,5),padding='same',activation='tanh'))
    model.add(tf.keras.layers.UpSampling2D(size=(2,2)))    #64*64像素的图像
    model.add(tf.keras.layers.Conv2D(3,(5,5),padding='Same',activation='tanh'))
    return model

#%%   构造一个 Sequential 对象，包含一个 生成器 和一个 判别器
# 输入=》生成器=》判别器=》输出

# Hyperparameter 超参数
EPOCH=100
BATCH_SIZE=128
LEARNING_RATE=0.0002
BETA_1=0.5


def generator_containing_discriminator(generator,discriminator):
    model=tf.keras.models.Sequential()
    model.add(generator)
    discriminator.trainable=False   #初始时 判别器时不可被训练的
    model.add(discriminator)
    return model











#%%

if __name__=='__main__':
    generator=generator_model()
    discriminator=discriminator_model()