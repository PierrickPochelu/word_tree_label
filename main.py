import numpy as np
import keras
from keras import backend as K
import tensorflow as tf
import os
import get_cifar10
import get_model
# implementation of CNN with simple softmax : https://keras.io/examples/cifar10_cnn/

epoch = 25
batch_size = 32


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def multi_optional_weighted_crossentropy(W):
    def optional_weighted_crossentropy(target, output, from_logits=False):
        coef_optional = tf.cast(tf.not_equal(target, -1), tf.float32)

        if not from_logits:
            output /= tf.reduce_sum(output,
                                    len(output.get_shape()) - 1,
                                    True)

            _epsilon = K.epsilon()
            output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)

            ce = tf.multiply(target * tf.log(output), coef_optional) * W
            r = - tf.reduce_sum(ce, len(output.get_shape()) - 1)
            return r
        else:
            return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                           logits=output)

    return optional_weighted_crossentropy





def compute_weights_for_CE(y):
    first_sample = 0
    nboutputs = len(y)


    # compute frequencies
    tree = []
    for i in range(nboutputs):  # for each softmax i
        y_output_i = y[i]
        softmax_size=len(y_output_i[first_sample])

        #compute value frequencies for i
        nb_minus_1 = np.zeros(softmax_size)
        nb_0 = np.zeros(softmax_size)
        nb_1 = np.zeros(softmax_size)
        for j in range(softmax_size):  # for each ouput j in softmax i
            nb_minus_1[j] = (y_output_i[:, j] == -1).sum()
            nb_0[j] = (y_output_i[:, j] == 0).sum()
            nb_1[j] = (y_output_i[:, j] == 1).sum()

        res_i = nb_1 / (nb_1 + nb_0)
        tree.append(res_i)

    # compute weights
    # scale=1./np.sum(tree[i],axis=0)
    for i in range(nboutputs):
        tree[i] = (1. / tree[i])

    # normalize weights
    tree = [softmax_weights / np.sum(softmax_weights) for softmax_weights in tree]

    return tree




def evaluate(model,xtest,ytest,ytest_multisoftmax):



    output=model.predict(xtest)
    pred1 = np.argmax(output[0],axis=1)
    pred2 = np.argmax(output[1],axis=1)
    pred3 = np.argmax(output[2],axis=1)
    y1=np.argmax(ytest_multisoftmax[0],axis=1)
    y2=np.argmax(ytest_multisoftmax[1],axis=1)
    y3=np.argmax(ytest_multisoftmax[2],axis=1)

    accuracy=get_cifar10.decode_from_multi_to_class(pred1,pred2,pred3,ytest)

    correct_pred1=np.mean( pred1[y1!=-1]==y1[y1!=-1] )
    correct_pred2=np.mean( pred2[y2!=-1]==y2[y2!=-1] )
    correct_pred3=np.mean( pred3[y3!=-1]==y3[y3!=-1]  )
    correct_overall=np.mean(accuracy)




    return correct_overall, correct_pred1, correct_pred2, correct_pred3

if __name__ == "__main__":
    xtrain,ytrain, ytrain_multisoftmax, xtest, ytest, ytest_multisoftmax = get_cifar10.get_cifar10_dataset()



    # multi optional softmax
    W = compute_weights_for_CE(ytrain_multisoftmax)
    multi_optional_softmax = [multi_optional_weighted_crossentropy(W[0]),
                              multi_optional_weighted_crossentropy(W[1]),
                              multi_optional_weighted_crossentropy(W[2])]
    print("multi optional softmax : cifar10")
    model = get_model.get_model(multi_optional_softmax)
    model.fit(x=xtrain, y=ytrain_multisoftmax, batch_size=batch_size, epochs=epoch, verbose=0)
    out=evaluate(model,xtest,ytest,ytest_multisoftmax)
    print(out)


    keras.backend.clear_session()
    print("cross entropy : cifar10")
    model = get_model.get_model(nb_output=10)
    model.fit(x=xtrain, y=ytrain, batch_size=batch_size, epochs=epoch, verbose=0)
    out=model.predict(xtest)
    #10 classes
    y_argmax=np.argmax(ytest,axis=1)
    out_argmax=np.argmax(out,axis=1)
    print(np.mean(y_argmax==out_argmax))
    # 2 classes
    print("cross entropy : animal or vehicle ? With 10 outputs")
    out_binary = get_cifar10.animal_or_vehicle(out)
    out_binary_argmax=out=np.argmax(out_binary,axis=1)
    ytest_binary = get_cifar10.animal_or_vehicle(ytest)
    ytest_binary_argmax=np.argmax(ytest_binary,axis=1)
    print(np.mean(ytest_binary_argmax==out_binary_argmax))


    keras.backend.clear_session()

    print("simple cross entropy : animal or vehicle ? with 2 output neural network")
    ytrain_binary=get_cifar10.animal_or_vehicle(ytrain)
    ytest_binary=get_cifar10.animal_or_vehicle(ytest)
    model = get_model.get_model(nb_output=2)
    model.fit(x=xtrain, y=np.array(ytrain_binary), batch_size=batch_size, epochs=epoch, verbose=0)
    y_argmax=np.argmax(ytest_binary,axis=1)
    out=np.argmax(model.predict(xtest),axis=1)
    print(np.mean(y_argmax==out))
