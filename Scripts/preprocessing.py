from scipy.io import loadmat, savemat
import numpy as np
from skimage.transform import resize
import sklearn

def minibatcher(X, y, d, batch_size=128):
    n_batches = int(np.ceil(len(X)/batch_size))
    X_batches = []
    y_batches = []
    d_batches = []
    for i in range(n_batches):
        try:
            X_batches.append(X[i*batch_size:(i+1)*batch_size,:,:,:])
            y_batches.append(y[i*batch_size:(i+1)*batch_size])
            d_batches.append(d[i*batch_size:(i+1)*batch_size])
        except:
            X_batches.append(X[i*batch_size:,:,:,:])
            y_batches.append(y[i*batch_size:])
            d_batches.append(d[i*batch_size:])
    return X_batches, y_batches, d_batches

if __name__ == "__main__":
    #target domain
    t_dm = loadmat(r'C:\Users\willr\Documents\University\FYP\Model\Data\MNIST\mnist-original.mat')
    t_dm_new = {}

    t_dm_X = t_dm["data"].reshape(28,28,t_dm["data"].shape[-1])
    t_dm_y = t_dm["label"]

    new_mnist = np.zeros((32,32,t_dm_X.shape[-1]))
    for i in range(t_dm_X.shape[-1]):
        new_mnist[:,:,i] = resize(t_dm_X[:,:,i], (32, 32))

    t_dm_X = new_mnist
    t_dm_X, t_dm_y = sklearn.utils.shuffle(np.moveaxis(t_dm_X,-1,0), 
                                           np.moveaxis(t_dm_y,-1,0))

    t_dm_X = np.moveaxis(np.repeat(t_dm_X[:,:,:,np.newaxis],3,-1),-1,1)
    t_dm_y = t_dm_y.astype(int)
    t_dm_d = np.ones(t_dm_y.shape).astype(int)

    t_dm_new["X_batches"], t_dm_new["y_batches"], t_dm_new["d_batches"] = minibatcher(t_dm_X, t_dm_y, t_dm_d)

    #source domain
    s_dm = loadmat(r'C:\Users\willr\Documents\University\FYP\Model\Data\SVHN\train_32x32.mat')
    s_dm_new = {}

    s_dm_X = s_dm["X"]/255
    s_dm_y = s_dm["y"]

    s_dm_X, s_dm_y = sklearn.utils.shuffle(np.moveaxis(np.moveaxis(s_dm_X,-1,0),-1,1), 
                                                       s_dm_y)
    s_dm_y[s_dm_y == 10] = 0    #because 10 labelled instead of 0... all positions in output vector moved one spot

    s_dm_X = s_dm_X
    s_dm_y = s_dm_y.astype(int)
    s_dm_d = np.zeros(s_dm_y.shape).astype(int)

    s_dm_new["X_batches"], s_dm_new["y_batches"], s_dm_new["d_batches"] = minibatcher(s_dm_X, s_dm_y, s_dm_d)

    savemat("mnist_processed.mat", t_dm_new)
    savemat("SVHN_processed.mat", s_dm_new)