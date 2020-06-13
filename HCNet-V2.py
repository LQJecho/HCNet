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
    y = Input(shape=(18,),dtype=tf.float32)
    index=Input(shape=(9,),dtype=tf.float32)
    pnr = Input(shape=(1,),dtype=tf.float32)
    tmp=concatenate([y,index,pnr],axis=1)
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
    model = Model([y,index,pnr], aod)
    model.compile(optimizer=tf.train.AdamOptimizer(), loss=RankMse)
    model.summary()
    return model

train_size=300000
dir = '../dataset/RF/8RF/train.mat'
y_c = sio.loadmat(dir)['Y']
train_y=np.real(y_c*np.conj(y_c))
train_index = sio.loadmat(dir)['Index']
train_label = sio.loadmat('../dataset/channel/channel_train.mat')['Rad']
train_pnr = sio.loadmat(dir)['PNR']
train_y=train_y[0:train_size,:]
train_index=train_index[0:train_size,:]
train_pnr=train_pnr[0:train_size,:]
train_label = train_label[0:train_size,:]

val_size=100000
dir = '../dataset/RF/8RF/val.mat'
y_c = sio.loadmat(dir)['Y']
val_y=np.real(y_c*np.conj(y_c))
val_index = sio.loadmat(dir)['Index']
val_pnr = sio.loadmat(dir)['PNR']
val_label = sio.loadmat('../dataset/channel/channel_val.mat')['Rad']
val_y=val_y[0:val_size,:]
val_index=val_index[0:val_size,:]
val_pnr=val_pnr[0:val_size,:]
val_label = val_label[0:val_size,:]

train_y=np.concatenate([train_y[:,24:30],train_y[:,54:60],train_y[:,84:90]],axis=1)
train_index=np.concatenate([train_index[:,12:15],train_index[:,27:30],train_index[:,42:45]],axis=1)
val_y=np.concatenate([val_y[:,24:30],val_y[:,54:60],val_y[:,84:90]],axis=1)
val_index=np.concatenate([val_index[:,12:15],val_index[:,27:30],val_index[:,42:45]],axis=1)

path = './model/HCNet-V2.h5'
checkpoint = ModelCheckpoint(path, monitor='val_loss', verbose=1, save_best_only=True,
                             mode='min', save_weights_only=True)
model = channel_estimation()
model.fit([train_y,train_index,train_pnr], train_label, batch_size=2048, epochs=100, verbose=2,
          callbacks=[checkpoint], validation_data=([val_y,val_index,val_pnr], val_label), shuffle=True)

model.load_weights(path)
test_size=10000
test_label=sio.loadmat('../dataset/channel/channel_test.mat')['Rad']
test_label=test_label[0:test_size,:]

# Fig4, curve HCNet-V2
MSE=[]
PNR_db=np.arange(-20,15,5)
for pnr_db in PNR_db:
    dir='../dataset/RF/8RF/pnr{pnr}.mat'
    dir=dir.format(pnr=pnr_db)
    test_index=sio.loadmat(dir)['Index']
    test_y_c=sio.loadmat(dir)['Y']
    test_y=np.real(test_y_c*np.conj(test_y_c))
    pnr = np.power(10, pnr_db / 10)
    test_pnr=pnr*np.ones([test_size,1])
    test_y=test_y[0:test_size,:]
    test_index=test_index[0:test_size,:]
    test_y = np.concatenate([test_y[:, 24:30], test_y[:, 54:60], test_y[:, 84:90]], axis=1)
    test_index = np.concatenate([test_index[:, 12:15], test_index[:, 27:30], test_index[:, 42:45]], axis=1)
    mse=model.evaluate([test_y,test_index,test_pnr],test_label,batch_size=test_size)
    MSE.append(mse)

print(MSE)