import numpy as np
import keras
from keras import backend as K
import tensorflow as tf


nf=np.random.uniform((0.5,-0.5),(1.5,0.5),(100,2)) #1,0
f1=np.random.uniform((-0.5,-0.5),(0.5,0.5),(100,2)) # 0,0
f2=np.random.uniform((-0.5,0.5),(0.5,1.5),(100,2)) # 0,1
f3=np.random.uniform((-0.5,1.5),(0.5,2.5),(100,2)) #0,2
xtrain=np.concatenate((nf,f1,f2,f3))


def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(tf.multiply(y_pred, tf.cast(tf.not_equal(y_true, -1), tf.float32)),
                                        tf.multiply(y_true, tf.cast(tf.not_equal(y_true, -1), tf.float32))), axis=-1)

class1= [np.array([1,0]),np.array([-1,-1,-1])]
class2= [np.array([0,1]),np.array([1,0,0])]
class3= [np.array([0,1]),np.array([0,1,0])]
class4= [np.array([0,1]),np.array([0,0,1])]
ynf=np.array([ class1 for i in range(100)])
yf1=np.array([ class2 for i in range(100)])
yf2=np.array([ class3 for i in range(100)])
yf3=np.array([ class4 for i in range(100)])
ytrain=np.concatenate((ynf,yf1,yf2,yf3))

ids=np.array(range(xtrain.shape[0]))
np.random.shuffle(ids)
xtrain=xtrain[ids]
ytrain=ytrain[ids]

input = keras.models.Input(shape=(2,))
x = keras.layers.Dense(5, activation='relu')(input)
output1 = keras.layers.Dense(2, activation='softmax')(x)
output2 = keras.layers.Dense(3, activation='softmax')(x)
model = keras.models.Model(inputs=input, outputs=[output1, output2])
model.compile(optimizer='adam',
              loss=['categorical_crossentropy', 'categorical_crossentropy'])


print(xtrain.shape)

ytrain_double=[]
ytrain_double.append( np.array([ysample[0]  for ysample in ytrain ]) )
ytrain_double.append( np.array([ysample[1]  for ysample in ytrain ]) )

assert(len(ytrain_double)==2)
assert(ytrain_double[0].shape==(400,2))
assert(ytrain_double[1].shape==(400,3))

model.fit(x=xtrain,y=ytrain_double,batch_size=8,epochs=100,verbose=2)

print(model.predict(np.array([[1,0]])))
print(model.predict(np.array([[0.6,0]])))

print(model.predict(np.array([[0.4,0]])))
print(model.predict(np.array([[0,1]])))
