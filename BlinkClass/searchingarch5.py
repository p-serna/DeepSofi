from numpy import *
from matplotlib.pylab import *
import random
import sys
import io
import os
import glob
import h5py


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D, Concatenate
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam
from keras.regularizers import L1L2
from keras import backend as K
from keras.layers import MaxPooling2D, AveragePooling2D,Conv2D
from keras.layers import Lambda
from keras.backend import transpose
from keras import regularizers


def get_random_time_segment(segment_frames,total_frames=12000):
    '''
    Gets a random time segment of duration segment_frames in a file
    with number of frames: total_frames
    '''
    
    segment_start = randint(0, high = total_frames-
                                   segment_frames)
    segment_end = segment_start + segment_frames
    
    return (segment_start, segment_end)
    
    
nframes = 1000 

# ~ print(X.shape, Y.shape)
    
pars = load("Training/roival.npy")

lpar = len(pars)
Xdev = []
Ydev = []
for i in range(lpar-200,lpar):
    xt = load("Training/roi"+str(i).zfill(4)+".npy")
    par = pars[i]
    for j in range(5):
        start,end = get_random_time_segment(nframes)
        xtt = xt[start:end,:].transpose()
        s0 = std(xtt)/mean(xtt)
        s1 = mean(xtt)
        xtt = xtt/s1
        s2 = std(xtt)
        s3 = mean((xtt-1.0)**3)/s2**3
        s4 = mean((xtt-1.0)**4)/s2**4
        s0t = array([s2,s3,s4,(s3**2+1/s4),0,0,0,0,0])
        
        xtt = column_stack((s0t,xtt))
        Xdev.append(xtt)
        #Ydev.append(concatenate((yt[start:end],par[-1:])))
        Ydev.append(par)
Xdev = array(Xdev)
Ydev = array(Ydev)

# ~ print(Xdev.shape, Ydev.shape)


Xdev = reshape(Xdev,(Xdev.shape[0],Xdev.shape[1],Xdev.shape[2],1))
Ydev = reshape(Ydev,(Ydev.shape[0],1,1,1))


def training(Xdev,Ydev,nf,nconv=20,epochs = 1000):
    nframes = 1000
    def model(input_shape,nf = 2,nconv = 20):
        '''
        Function used to create the model's graph in Keras
        
        Argument:
        -- input_shape. Shape of the model's input data (Keras conventions?!)
        
        Returns:
        -- model. Keras model instance
        '''
        tf_session = K.get_session()
        
        X_input = Input(shape = input_shape)
        
        # Layers

        X = X_input
        
        Xa = Lambda(lambda x: x[:,:,1:,:], output_shape=(input_shape[0],input_shape[1]-1,input_shape[2]))(X)
        X  = Lambda(lambda x: x[:,:4,:1,:], output_shape=(4,1,1))(X)
        X = Reshape((1,1,4))(X)
     
        Xashape = array(input_shape)
        Xashape[-1] -= 1 
        n1, n2, s = (20,10,4)

        Xa = Dropout(0.2)(Xa)
        Xb = Conv2D(nconv,(25,40),strides=(1,5),padding="valid")(Xa)
        Xc = AveragePooling2D((25,40),strides=(1,5),padding="valid")(Xa)
        Xa = MaxPooling2D((25,40),strides=(1,5),padding="valid")(Xa)
        
        Xa = Concatenate()([Xa,Xb,Xc])
        Xa = BatchNormalization()(Xa)   

        Xa = Reshape((1,1,193*(nconv+2)))(Xa)
        
        Xa = Dropout(0.2)(Xa)
        Xa = Dense(nf,activation="sigmoid")(Xa)
        #Xa = BatchNormalization()(Xa)
            

        X = Concatenate(axis=3)([X,Xa])
        X = Dense(1,activation="sigmoid")(X)

        #X = Xa
        # Defining the model
        
        model = Model(inputs = X_input, outputs = X)
        
        return(model)
        
    model = model(input_shape = (25,1+nframes,1),nf = nf,nconv = nconv)
    opt = Adam(lr=0.001, beta_1=0.95, beta_2=0.999, decay=0.001)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    pars = load("Training/roival.npy")

    lpar = len(pars)

    X = []
    Y = []
    for i in permutation(lpar-200):
        xt = load("Training/roi"+str(i).zfill(4)+".npy")
        par = pars[i]
        nj = 3
        if par == 1: nj = 8
        for j in range(nj):
            start,end = get_random_time_segment(nframes)
            xtt = xt[start:end,:].transpose()
            s0 = std(xtt)/mean(xtt)
            s1 = mean(xtt)
            xtt = xtt/s1
            s2 = std(xtt)
            s3 = mean((xtt-1.0)**3)/s2**3
            s4 = mean((xtt-1.0)**4)/s2**4
            s0t = array([s2,s3,s4,(s3**2+1/s4),0,0,0,0,0])
            
            xtt = column_stack((s0t,xtt))
            X.append(xtt)
            #Y.append(concatenate((yt[start:end],par[-1:])))
            Y.append(par)
    X = array(X)
    Y = array(Y)
    X = reshape(X,(X.shape[0],X.shape[1],X.shape[2],1))
    Y = reshape(Y,(Y.shape[0],1,1,1))

    history1 = model.fit(X, Y, batch_size = 500, epochs = epochs)
         
    lpar = len(pars)

    nframes = 1000 

    pars = load("/export/home1/users/bssn/serna/Downloads/Classifier/fullimage/moviefull/roival.npy")
    pars = pars[:,2]
    lpar = len(pars)

    X = []
    Y = []
    for i in permutation(lpar-200):
        xt = load("/export/home1/users/bssn/serna/Downloads/Classifier/fullimage/moviefull/roi_F01A"+str(i).zfill(4)+".npy")
        par = pars[i]
        nj = 3
        if par == 1: nj = 3
        for j in range(nj):
            start,end = get_random_time_segment(nframes)
            xtt = xt[start:end,:].transpose()
            s0 = std(xtt)/mean(xtt)
            s1 = mean(xtt)
            xtt = xtt/s1
            s2 = std(xtt)
            s3 = mean((xtt-1.0)**3)/s2**3
            s4 = mean((xtt-1.0)**4)/s2**4
            s0t = array([s2,s3,s4,(s3**2+1/s4),0,0,0,0,0])
            
            xtt = column_stack((s0t,xtt))
            X.append(xtt)
            #Y.append(concatenate((yt[start:end],par[-1:])))
            Y.append(par)
    X = array(X)
    Y = array(Y)
    X = reshape(X,(X.shape[0],X.shape[1],X.shape[2],1))
    Y = reshape(Y,(Y.shape[0],1,1,1))
    history2 = model.fit(X, Y, batch_size = 500, epochs = epochs)
            
    loss, acc = model.evaluate(Xdev, Ydev)

    return(loss,acc,model)
    

nfs = [2,4,6,8,10,20,50]
nconvs = [5,10,20,40,80]

pcond = [ [nf,nconv] for nf in nfs for nconv in nconvs]

res = []
for i, np in enumerate(pcond):
    nf,nconv = np
    loss,acc,_  = training(Xdev,Ydev,nf=nf,nconv=nconv,epochs = 500)
    print(" Training with",nf,nconv,"gives",acc)
    res.append([nf,nconv,loss,acc])
    savetxt("temp.dat",array(res))

savetxt("tempF.dat",array(res))


nfo = 4
nconvo = 10
loss,acc,model = training(Xdev,Ydev,nf=nfo,nconv=nconvo,epochs = 1000)

import time
date = time.localtime()
dates = str(date.tm_year)+str(date.tm_mon)+str(date.tm_mday)+'_'+str(date.tm_hour)

model.save("classifier"+dates+".h5")
