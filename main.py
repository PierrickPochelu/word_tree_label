import numpy as np
import keras
from keras import backend as K
import tensorflow as tf

epoch=100
batch_size=8
split=0.5
nb_experiments=10

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


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

def reformat_to_multi_output(y):
    y_double=[]
    y_double.append( np.array([ysample[0]  for ysample in y ]) )
    y_double.append( np.array([ysample[1]  for ysample in y ]) )
    return y_double


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

def compute_weights_for_CE(y):
    first_sample=0
    nboutputs=y.shape[1]
    softmax_size=[]
    for i in range(nboutputs):
        nb_class_in_this_out=y[first_sample][i].shape[0]
        softmax_size.append(nb_class_in_this_out)


    # compute frequencies
    tree=[]
    for i in range(nboutputs): # for each softmax i
        y_output_i=np.array(list(y[:,i]))

        nb_minus_1=np.zeros(softmax_size[i])
        nb_0=np.zeros(softmax_size[i])
        nb_1 = np.zeros(softmax_size[i])
        for j in range(softmax_size[i]): # for each ouput j in softmax i
            nb_minus_1[j]=(y_output_i[:,j]==-1).sum()
            nb_0[j] = (y_output_i[:,j] == 0).sum()
            nb_1[j] = (y_output_i[:,j] == 1).sum()

        res_i=nb_1/(nb_1+nb_0)
        tree.append( res_i )

    # compute weights
    #scale=1./np.sum(tree[i],axis=0)
    for i in range(nboutputs):
        tree[i] = (1./tree[i])

    return tree


def get_synthetic_dataset():
    # create X
    nf=np.random.uniform((0.5,-0.5),(1.5,0.5),(900,2)) #1,0
    f1=np.random.uniform((-0.5,-0.5),(0.5,0.5),(100,2)) # 0,0
    f2=np.random.uniform((-0.5,0.5),(0.5,1.5),(100,2)) # 0,1
    f3=np.random.uniform((-0.5,1.5),(0.5,2.5),(100,2)) #0,2
    x=np.concatenate((nf,f1,f2,f3))

    # create Y
    class1= [np.array([1,0]),np.array([-1,-1,-1])]
    class2= [np.array([0,1]),np.array([1,0,0])]
    class3= [np.array([0,1]),np.array([0,1,0])]
    class4= [np.array([0,1]),np.array([0,0,1])]
    ynf=np.array([ class1 for i in range(900)])
    yf1=np.array([ class2 for i in range(100)])
    yf2=np.array([ class3 for i in range(100)])
    yf3=np.array([ class4 for i in range(100)])
    y=np.concatenate((ynf,yf1,yf2,yf3))

    # shuffle
    ids=np.array(range(x.shape[0]))
    np.random.shuffle(ids)
    y=y[ids]

    # split train/test
    split_id=int(x.shape[0]*split)
    xtest=x[:split_id]
    xtrain=x[split_id:]
    ytest=y[:split_id]
    ytrain=y[split_id:]

    weights=compute_weights_for_CE(ytrain) # Warning : keras does not understand weights in multi softmax context

    # reformat Y
    ytrain_double=reformat_to_multi_output(ytrain)
    ytest_double=reformat_to_multi_output(ytest)



    return xtrain, ytrain_double, xtest, ytest_double,weights


if __name__=="__main__":
    xtrain, ytrain_double, xtest, ytest_double, w = get_synthetic_dataset()
    print(w)

    assert(len(ytrain_double)==2)
    assert(ytrain_double[0].shape[1]==2)
    assert(ytrain_double[1].shape[1]==3)

    # all together
    ev=np.array([0.,0.,0.],dtype=float)
    for i in range(nb_experiments):
        model = get_model(True, True)
        model.fit(x=xtrain,y=ytrain_double,batch_size=batch_size,epochs=epoch,verbose=0)
        ev+=model.evaluate(xtest,ytest_double)
    print(ev/nb_experiments)


    # loss 1 only
    ev=0
    for i in range(nb_experiments):
        model = get_model(True, False)
        model.fit(x=xtrain,y=ytrain_double[0],batch_size=batch_size,epochs=epoch,verbose=0)
        ev+=model.evaluate(xtest,ytest_double[0])
    print(ev/nb_experiments)

    # loss 2 only
    ev=0
    for i in range(nb_experiments):
        model = get_model(False, True)
        model.fit(x=xtrain,y=ytrain_double[1],batch_size=batch_size,epochs=epoch,verbose=0)
        ev+=model.evaluate(xtest,ytest_double[1])
    print(ev/nb_experiments)
