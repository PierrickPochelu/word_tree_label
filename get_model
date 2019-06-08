from keras import Input, Model, optimizers
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
import numpy as np

def get_model(multi_softmax=None,input_shape=(32,32,3),nb_output=10):


    input= Input(shape=input_shape)
    x=Conv2D(32, (3, 3), padding='same')(input)
    x=Activation('relu')(x)
    x=Conv2D(32, (3, 3))(x)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    x=Dropout(0.25)(x)
    x=Conv2D(64, (3, 3), padding='same')(x)
    x=Activation('relu')(x)
    x=Conv2D(64, (3, 3))(x)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    x=Dropout(0.25)(x)
    x=Flatten()(x)
    x=Dense(512)(x)
    x=Activation('relu')(x)
    x=Dropout(0.5)(x)

    #opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)
    opt = optimizers.Adam()

    if multi_softmax is not None:
        output1=Activation('softmax')(Dense(2)(x))
        output2=Activation('softmax')(Dense(6)(x))
        output3=Activation('softmax')(Dense(4)(x))

        model = Model(inputs=input, outputs=[output1, output2,output3])
        model.compile(optimizer=opt,
                      loss=multi_softmax)
    else:
        output=Activation('softmax')(Dense(nb_output)(x))
        model = Model(inputs=input, outputs=output)
        model.compile(optimizer=opt,
                      loss='categorical_crossentropy')


    return model
