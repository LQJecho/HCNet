from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import Input, Model
import tensorflow as tf
import numpy as np
import math
import scipy.io as sio

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.keras.backend.set_session(session)

def RankMse(y_true , y_pred):
    path1_pred = tf.slice(y_pred, [0, 0], [-1, 1])
    path2_pred = tf.slice(y_pred, [0, 1], [-1, 1])
    path3_pred = tf.slice(y_pred, [0, 2], [-1, 1])
    y_pred2=tf.concat([path1_pred,path3_pred,path2_pred],axis=1)
    y_pred3 = tf.concat([path2_pred, path1_pred, path3_pred],axis=1)
    y_pred4 = tf.concat([path2_pred, path3_pred, path1_pred],axis=1)
    y_pred5 = tf.concat([path3_pred, path1_pred, path2_pred],axis=1)
    y_pred6 = tf.concat([path3_pred, path2_pred, path1_pred],axis=1)
    mse_array=tf.concat([tf.reduce_mean(tf.square(y_true - y_pred),axis=1,keepdims=True),tf.reduce_mean(tf.square(y_true - y_pred2),axis=1,keepdims=True),
               tf.reduce_mean(tf.square(y_true - y_pred3),axis=1,keepdims=True),tf.reduce_mean(tf.square(y_true - y_pred4),axis=1,keepdims=True),
               tf.reduce_mean(tf.square(y_true - y_pred5),axis=1,keepdims=True),tf.reduce_mean(tf.square(y_true - y_pred6),axis=1,keepdims=True)],axis=1)
    rankmse=tf.reduce_min(mse_array, axis=1, keepdims=True)
    return tf.reduce_mean(rankmse)


def To_rad(tmp):
    return tmp*math.pi


def dense_unit_dropout(input_tensor, nn, rate):
    out_tensor = Dense(nn)(input_tensor)
    out_tensor = BatchNormalization()(out_tensor)
    out_tensor = tf.nn.relu(out_tensor)
    out_tensor = Dropout(rate)(out_tensor)
    return out_tensor


def channel_estimation():
    rate=0.1
    y = Input(shape=(90,),dtype=tf.float32)
    index=Input(shape=(45,),dtype=tf.float32)
    tmp=concatenate([y,index],axis=1)
    tmp=dense_unit_dropout(tmp,256,rate)
    tmp1 = dense_unit_dropout(tmp,32,rate)
    tmp2 = dense_unit_dropout(tmp, 32, rate)
    tmp3 = dense_unit_dropout(tmp, 32, rate)
    tmp1 = dense_unit_dropout(tmp1, 8, rate)
    tmp2 = dense_unit_dropout(tmp2, 8, rate)
    tmp3 = dense_unit_dropout(tmp3, 8, rate)
    tmp1=Dense(1, activation='sigmoid')(tmp1)
    tmp2 = Dense(1, activation='sigmoid')(tmp2)
    tmp3 = Dense(1, activation='sigmoid')(tmp3)
    tmp1=Lambda(To_rad)(tmp1)
    tmp2=Lambda(To_rad)(tmp2)
    tmp3=Lambda(To_rad)(tmp3)
    aod = concatenate([tmp1, tmp2,tmp3], axis=1)
    model = Model([y,index], aod)
    model.compile(optimizer=tf.train.AdamOptimizer(), loss=RankMse)
    model.summary()
    return model

train_size=300000
dir = '../dataset/Qbit/Q1train.mat'
y_c = sio.loadmat(dir)['Y']
train_y=np.real(y_c*np.conj(y_c))
train_index = sio.loadmat(dir)['Index']
train_label = sio.loadmat('../dataset/channel/channel_train.mat')['Rad']
train_y=train_y[0:train_size,:]
train_index=train_index[0:train_size,:]
train_label = train_label[0:train_size,:]

val_size=100000
dir = '../dataset/Qbit/Q1val.mat'
y_c = sio.loadmat(dir)['Y']
val_y=np.real(y_c*np.conj(y_c))
val_index = sio.loadmat(dir)['Index']
val_label = sio.loadmat('../dataset/channel/channel_val.mat')['Rad']
val_y=val_y[0:val_size,:]
val_index=val_index[0:val_size,:]
val_label = val_label[0:val_size,:]

path = './model/Qbit_1.h5'
checkpoint = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True,
                             mode='min', save_weights_only=True)
model = channel_estimation()
# model.fit([train_y,train_index], train_label, batch_size=2048, epochs=100, verbose=2,
#           callbacks=[checkpoint], validation_data=([val_y,val_index], val_label), shuffle=True)

model.load_weights(path)
test_size=10000
test_label=sio.loadmat('../dataset/channel/channel_test.mat')['Rad']
test_label=test_label[0:test_size,:]
Qbit=np.arange(1,8,1)

# Fig6, curve6, q_tr=1
MSE = []
for Q in Qbit:
    dir = '../dataset/Qbit/pnr0/Q{Q}.mat'
    dir = dir.format(Q=Q)
    test_index = sio.loadmat(dir)['Index']
    test_y_c = sio.loadmat(dir)['Y']
    test_y = np.real(test_y_c * np.conj(test_y_c))
    test_y = test_y[0:test_size, :]
    test_index = test_index[0:test_size, :]
    mse = model.evaluate([test_y, test_index], test_label, batch_size=test_size,verbose=0)
    MSE.append(mse)
print(MSE)