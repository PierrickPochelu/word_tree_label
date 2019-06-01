import numpy as np
import keras
from keras import backend as K
import tensorflow as tf

epoch=200
batch_size=8

# create X
nf=np.random.uniform((0.5,-0.5),(1.5,0.5),(100,2)) #1,0
f1=np.random.uniform((-0.5,-0.5),(0.5,0.5),(100,2)) # 0,0
f2=np.random.uniform((-0.5,0.5),(0.5,1.5),(100,2)) # 0,1
f3=np.random.uniform((-0.5,1.5),(0.5,2.5),(100,2)) #0,2
xtrain=np.concatenate((nf,f1,f2,f3))


def optional_categorical_crossentropy(target, output, from_logits=False):
    A=tf.cast(tf.not_equal(target, -1), tf.float32)

    if not from_logits:
        output /= tf.reduce_sum(output,
                                len(output.get_shape()) - 1,
                                True)
        _epsilon = K.epsilon()
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)

        ce=tf.multiply(target * tf.log(output),A)
        r=- tf.reduce_sum(ce,
                               len(output.get_shape()) - 1)
        return r
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                       logits=output)

# create Y
class1= [np.array([1,0]),np.array([-1,-1,-1])]
class2= [np.array([0,1]),np.array([1,0,0])]
class3= [np.array([0,1]),np.array([0,1,0])]
class4= [np.array([0,1]),np.array([0,0,1])]
ynf=np.array([ class1 for i in range(100)])
yf1=np.array([ class2 for i in range(100)])
yf2=np.array([ class3 for i in range(100)])
yf3=np.array([ class4 for i in range(100)])
ytrain=np.concatenate((ynf,yf1,yf2,yf3))

# shuffle
ids=np.array(range(xtrain.shape[0]))
np.random.shuffle(ids)
xtrain=xtrain[ids]
ytrain=ytrain[ids]

split=0.5
split_id=int(xtrain.shape[0]*split)
xtest=xtrain[:split_id]
xtrain=xtrain[split_id:]
ytest=ytrain[:split_id]
ytrain=ytrain[split_id:]



def get_model(is_out1=True, is_out2=True):
    input = keras.models.Input(shape=(2,))
    x = keras.layers.Dense(5, activation='relu')(input)
    if is_out1:
        output1 = keras.layers.Dense(2, activation='softmax')(x)
    if is_out2:
        output2 = keras.layers.Dense(3, activation='softmax')(x)


    if is_out1 and is_out2:
        model = keras.models.Model(inputs=input, outputs=[output1, output2])
        model.compile(optimizer='adam',
                      loss=[optional_categorical_crossentropy, optional_categorical_crossentropy])
    elif is_out1:
        model = keras.models.Model(inputs=input, outputs=[output1])
        model.compile(optimizer='adam',
                      loss=[optional_categorical_crossentropy])
    elif is_out2:
        model = keras.models.Model(inputs=input, outputs=[output2])
        model.compile(optimizer='adam',
                      loss=[optional_categorical_crossentropy])
    else:
        raise ValueError("No is_out")


    return model

print(xtrain.shape)

# reformat Y
ytrain_double=[]
ytrain_double.append( np.array([ysample[0]  for ysample in ytrain ]) )
ytrain_double.append( np.array([ysample[1]  for ysample in ytrain ]) )
ytest_double=[]
ytest_double.append( np.array([ysample[0]  for ysample in ytest ]) )
ytest_double.append( np.array([ysample[1]  for ysample in ytest ]) )


assert(len(ytrain_double)==2)
assert(ytrain_double[0].shape[1]==2)
assert(ytrain_double[1].shape[1]==3)

# all together
model = get_model(True, True)
model.fit(x=xtrain,y=ytrain_double,batch_size=batch_size,epochs=epoch,verbose=0
          )
print(model.evaluate(xtest,ytest_double))

# loss 1 only
model = get_model(True, False)
model.fit(x=xtrain,y=ytrain_double[0],batch_size=batch_size,epochs=epoch,verbose=0)
print(model.evaluate(xtest,ytest_double[0]))

# loss 2 only
model = get_model(False, True)
model.fit(x=xtrain,y=ytrain_double[1],batch_size=batch_size,epochs=epoch,verbose=0)
print(model.evaluate(xtest,ytest_double[1]))
