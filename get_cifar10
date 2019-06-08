import pickle
import numpy as np
import os
from urllib.request import urlretrieve
import tarfile
import zipfile
import sys


def get_data_set(name="train"):
    x = None
    y = None

    __maybe_download_and_extract()

    folder_name = "cifar_10"

    f = open('./data_set/'+folder_name+'/batches.meta', 'rb')
    f.close()

    if name is "train":
        for i in range(5):
            f = open('./data_set/'+folder_name+'/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 32, 32, 3])
            #_X = _X.transpose([0, 2, 3, 1])
            #_X = _X.reshape(-1, 32*32*3)

            if x is None:
                x = _X
                y = _Y
            else:
                x = np.concatenate((x, _X), axis=0)
                y = np.concatenate((y, _Y), axis=0)

    elif name is "test":
        f = open('./data_set/'+folder_name+'/test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        x = datadict["data"]
        y = np.array(datadict['labels'])

        x = np.array(x, dtype=float) / 255.0
        x = x.reshape([-1, 32, 32, 3])
        #x = x.transpose([0, 2, 3, 1])
        #x = x.reshape(-1, 32*32*3)

    return x, __dense_to_one_hot(y)


def __dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


def _print_download_progress(count, block_size, total_size):
    pct_complete = float(count * block_size) / total_size
    msg = "\r- Download progress: {0:.1%}".format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()


def __maybe_download_and_extract():
    main_directory = "./data_set/"
    cifar_10_directory = main_directory+"cifar_10/"
    if not os.path.exists(main_directory):
        os.makedirs(main_directory)

        url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar_10 = file_path
        file_path, _ = urlretrieve(url=url, filename=file_path, reporthook=_print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")

        os.rename(main_directory+"./cifar-10-batches-py", cifar_10_directory)
        os.remove(zip_cifar_10)


###################################################################################################
def reformat_to_multi_output(y):
    y_multi_softmax=[]
    nb=len(y[0])
    for i in range(nb):
        y_multi_softmax.append( np.array([ysample[i]  for ysample in y ]) )
    return y_multi_softmax

def get_cifar10_dataset():
    xtrain, ytrain = get_data_set("train")
    xtest, ytest = get_data_set("test")

    # reformat Y
    ytrain_double = reformat_cifar10(ytrain)
    ytest_double = reformat_cifar10(ytest)

    # reformat to multi softmax
    ytrain_double = reformat_to_multi_output(ytrain_double)
    ytest_double = reformat_to_multi_output(ytest_double)



    return xtrain, ytrain, ytrain_double, xtest, ytest, ytest_double

def reformat_cifar10(y):
    """
    :param y: array 1D of class id from 0 to 9
    :return: path expected in decision. special value "-1" means we doesn't care this value and disable backprop.
    result is a list of list containing respectively 2,6,4 elements.
    """
    dict = {
        0: [np.array([0, 1]), np.array([-1, -1, -1, -1, -1, -1]), np.array([1, 0, 0, 0])],
        1: [np.array([0, 1]), np.array([-1, -1, -1, -1, -1, -1]), np.array([0, 1, 0, 0])],
        2: [np.array([1, 0]), np.array([1, 0, 0, 0, 0, 0]), np.array([-1, -1, -1, -1])],
        3: [np.array([1, 0]), np.array([0, 1, 0, 0, 0, 0]), np.array([-1, -1, -1, -1])],
        4: [np.array([1, 0]), np.array([0, 0, 1, 0, 0, 0]), np.array([-1, -1, -1, -1])],
        5: [np.array([1, 0]), np.array([0, 0, 0, 1, 0, 0]), np.array([-1, -1, -1, -1])],
        6: [np.array([1, 0]), np.array([0, 0, 0, 0, 1, 0]), np.array([-1, -1, -1, -1])],
        7: [np.array([1, 0]), np.array([0, 0, 0, 0, 0, 1]), np.array([-1, -1, -1, -1])],
        8: [np.array([0, 1]), np.array([-1, -1, -1, -1, -1, -1]), np.array([0, 0, 1, 0])],
        9: [np.array([0, 1]), np.array([-1, -1, -1, -1, -1, -1]), np.array([0, 0, 0, 1])],
    }

    def reformat_this_sample(sample_y):
        return dict[sample_y]

    y_argmax=np.argmax(y,axis=1)
    return [reformat_this_sample(sample_y) for sample_y in y_argmax]




def decode_from_multi_to_class(output1,output2,output3,y):
    assert(len(output1)==len(output2)==len(output3))

    def format_this_sample(o1,o2,o3):
        res=0
        if o1==0:
            if o2==0:
                return 2
            elif o2==1:
                return 3
            elif o2==2:
                return 4
            elif o2==3:
                return 5
            elif o2==4:
                return 6
            elif o2==5:
                return 7
            else :
                raise ValueError("Animal not understood")
        elif o1==1:
            if o3==0:
                return 0
            elif o3==1:
                return 1
            elif o3==2:
                return 8
            elif o3==3:
                return 9
            else :
                raise ValueError("Vehicle not understood")
        else:
            raise ValueError("Not understood")

    correct=np.zeros((output1.shape[0],))
    y_argmax=np.argmax(y,axis=1)
    for i in range(len(output1)):
        o=format_this_sample(output1[i],output2[i],output3[i])
        if o==y_argmax[i]:
            correct[i]=1

    return correct

def animal_or_vehicle(y):
    y_binary=np.zeros((y.shape[0],2))
    y_argmax=np.argmax(y, axis=1)
    for i in range(y.shape[0]):
        if y_argmax[i] in [0,1,8,9]:
            y_binary[i,1]=1
        else:
            y_binary[i,0]=1
    return y_binary

def which_animal_is_it(y):

    y_argmax=np.argmax(y, axis=1)

    def f(arg_id):
        v = [2,3,4,5,6,7]
        return arg_id in v

    y_argmax=list(filter(f,y_argmax))

    y_binary = np.zeros((len(y_argmax), 6))
    for i in range(len(y_argmax)):
        if y_argmax[i] == 2:
            y_binary[i,0]=1
        elif y_argmax[i] == 3:
            y_binary[i,1]=1
        elif y_argmax[i] == 4:
            y_binary[i,2]=1
        elif y_argmax[i] == 5:
            y_binary[i,3]=1
        elif y_argmax[i] == 6:
            y_binary[i,4]=1
        elif y_argmax[i] == 7:
            y_binary[i,5]=1
    return y_binary

def which_vehicle_is_it(y):
    y_argmax=np.argmax(y, axis=1)
    def f(arg_id):
        v = [0,1,8,9]
        return arg_id in v
    y_argmax = list(filter(f,y_argmax))

    y_binary=np.zeros((len(y_argmax),4))
    for i in range(len(y_argmax)):
        if y_argmax[i] == 0:
            y_binary[i,0]=1
        elif y_argmax[i] == 1:
            y_binary[i,1]=1
        elif y_argmax[i] == 8:
            y_binary[i,2]=1
        elif y_argmax[i] == 9:
            y_binary[i,3]=1
    return y_binary
