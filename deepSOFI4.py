from numpy import *
from matplotlib.pylab import *
import random
import sys
import io
import os
import glob
import h5py
import IPython
from td_utils import *



from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D, Concatenate
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.layers import MaxPooling2D, MaxPooling3D, AveragePooling2D, AveragePooling3D
from keras.layers import Conv2D,Conv3D, Lambda
from keras.backend import transpose
from keras.optimizers import Adam
from keras import regularizers


i = randint(0,900)
home = "/export/home1/users/bssn/serna"
datadir = home+"/SyntheticData/syndat/"
Ft = load(datadir+"d"+str(i).zfill(4)+".npy")

Ftsh = Ft.shape
width,height,T = Ftsh  
print(Ftsh)
t = arange(Ftsh[-1])

def get_random_time_segment(segment_frames,total_frames=12000):
    '''
    Gets a random time segment of duration segment_frames in a file
    with number of frames: total_frames
    '''
    
    segment_start = randint(0, high = total_frames-
                                   segment_frames)
    segment_end = segment_start + segment_frames
    
    return (segment_start, segment_end)
    
def is_overlapping(segment_time, previous_segments):
    '''
    This function checks if the time of a segment overlaps with the
    existing segments.
    '''
    s_start, s_end = segment_time
    
    overlap = False
    
    for prev_start, prev_end in previous_segments:
        if s_start <= prev_end and s_end >= prev_start:
            overlap = True
    
    return overlap
    
from scipy.fftpack import fft2,fftshift,ifftshift,ifft2

def ftaugment(img,magn=2):
    sh = img.shape
    sh2 = (sh*(magn-1))
    sh2 = ((sh2[0])//2,(sh2[1])//2)
    fftim  = fftshift(fft2(img))
    fftim  =  pad(fftim,(sh2[0],sh2[1]),'constant')    
    imgn = real(ifft2(ifftshift(fftim)))
    return(imgn)

def ftvaugment(img,magn=2):
    sh = img.shape
    sh2 = array([sh[0],sh[1]])*(magn-1)
    sh2 = ((sh2[0])//2,(sh2[0])//2)
    fftim  = fftshift(fft2(img,axes=(0,1)),axes=(0,1))
    fftim  =  pad(fftim,(sh2[0],sh2[1]),'constant') 
    imgn = real(ifft2(ifftshift(fftim,axes=(0,1)),axes=(0,1)))
    imgn = imgn[:,:,sh2[0]:-sh2[0]]
    return(imgn)
    
    
# 1 segundo. This will be used as another (meta)parameter, 
# which we want to decrease
nframes = 500
magn=4

X = []
Y = []
X = zeros((800,width*magn,height*magn,nframes))
Y = zeros((800,width*magn,height*magn))

for i in range(800):
    xt = load(datadir+"d"+str(i).zfill(4)+".npy")
    yt = load(datadir+"o"+str(magn)+"_"+str(i).zfill(4)+".npy") 
    xt = xt/mean(xt.flatten())
    xt = ftvaugment(xt,magn)
    ymax = max(1,max(yt.flatten()))
    yt = yt
    #par = pars[i]
    #xt = reshape(xt,(width*height,T))
    #yt = reshape(yt,(4*width*height))
    for j in range(1):
        start,end = get_random_time_segment(nframes,T)
        X[i,:,:,:] = xt[:,:,start:end]
        #Y.append(concatenate((yt[start:end],par[-1:])))
        #ytr = 0.05*ymax*randn(magn*width,magn*height)
        Y[i,:,:] = (yt)/ymax
    if i %200 == 0:
        print("Iteracion #",i)
X = array(X)
Y = array(Y)
print(X.shape, Y.shape)
    

Xdev = []
Ydev = []
Xdev = zeros((800,width*magn,height*magn,nframes))
Ydev = zeros((800,width*magn,height*magn))
for i in range(800,900):
    xt = load(datadir+"d"+str(i).zfill(4)+".npy")
    yt = load(datadir+"o"+str(magn)+"_"+str(i).zfill(4)+".npy")
    xt = xt/mean(xt.flatten())
    xt = ftvaugment(xt,magn)
    
    ymax = max(1,max(yt.flatten()))
    yt = yt
    #par = pars[i]
    #xt = reshape(xt,(width*height,T))
    #yt = reshape(yt,(4*width*height))
    for j in range(1):
        start,end = get_random_time_segment(nframes,T)
        X[i,:,:,:] = xt[:,:,start:end]
        #Y.append(concatenate((yt[start:end],par[-1:])))
        #ytr = 0.05*ymax*randn(magn*width,magn*height)
        Y[i,:,:] = (yt)/ymax
Xdev = array(Xdev)
Ydev = array(Ydev)

print(Xdev.shape, Ydev.shape)


X = reshape(X,(X.shape[0],X.shape[1],X.shape[2],X.shape[3],1))
Y = reshape(Y,(Y.shape[0],Y.shape[1],Y.shape[2],1,1))

Xdev = reshape(Xdev,(Xdev.shape[0],Xdev.shape[1],Xdev.shape[2],Xdev.shape[3],1))
Ydev = reshape(Ydev,(Ydev.shape[0],Ydev.shape[1],Ydev.shape[2],1,1))

def model(input_shape):
    '''
    Function used to create the model's graph in Keras
    
    Argument:
    -- input_shape. Shape of the model's input data (Keras conventions?!)
    
    Returns:
    -- model. Keras model instance
    '''

    X_input = Input(shape = input_shape)
    
    w,h,T,_ = input_shape
    # Layers

    X = X_input
    
    Xa = MaxPooling3D((1,1,40),strides=(1,1,5),padding="same")(X)
    Xb = AveragePooling3D((1,1,40),strides=(1,1,5),padding="same")(X)
    Xa2 =  Lambda(lambda x: x * x)(Xa)
    Xb2 = Lambda(lambda x: x * x)(Xb)
    Xa3 =  Lambda(lambda x: x **3)(Xa)
    Xb3 = Lambda(lambda x: x **3)(Xb)
    Xa4 =  Lambda(lambda x: x **4)(Xa)
    Xb4 = Lambda(lambda x: x **4)(Xb)

    X = Concatenate()([Xa,Xb,Xa2,Xb2,Xa3,Xb3,Xa4,Xb4])
    X = BatchNormalization()(X)   

    X = Dropout(0.3)(X)
    
    X = Conv3D(80,(4,4,8),strides=(1,1,3),padding="same")(X)
    X = Activation("relu")(X)
    X = BatchNormalization()(X)
    
    X = Dropout(0.3)(X)
    X = AveragePooling3D((1,1,20),strides=(1,1,3),padding="same")(X)
    X = BatchNormalization()(X)   
   
    X = Dropout(0.3)(X)
    X = Conv3D(40,(1,1,12),strides=(1,1,12),padding="valid")(X)
    X = Activation("relu")(X)
    X = BatchNormalization()(X) 

    
    #X = Dropout(0.3)(X)
    #X = MaxPooling3D((1,1,10),strides=(1,1,1),padding="same")(X)
    #X = Activation("relu")(X)
    #X = BatchNormalization()(X) 

    #X = Dropout(0.3)(X)
    #X = Conv3D(80,(1,1,10),strides=(1,1,2),padding="valid")(X)
    #X = Activation("relu")(X)
    #X = BatchNormalization()(X) 
    X = Dropout(0.3)(X)
    X = Dense(20,activation="relu")(X)
    X = BatchNormalization()(X) 

    X = Dropout(0.3)(X)
    X = Dense(5,activation="relu")(X)
    X = BatchNormalization()(X) 
    
    
    X = Dropout(0.3)(X)
    X = Dense(1,activation="relu")(X)

    # Defining the model
    
    model = Model(inputs = X_input, outputs = X)
    
    return model
    
model = model(input_shape = (magn*width,magn*height,nframes,1))
model.summary()

from keras.optimizers import SGD

import keras.backend as K

#Mendel ...
def MOC(y_true,y_pred):
    '''Just another crossentropy'''
    ly = K.sum(y_true*y_pred)
    ly2 = K.sum(y_pred*y_pred)
    ly3 = K.sum(y_true*y_true)
    out = ly/K.sqrt(ly2*ly3)
    return out

def MOCl(y_true,y_pred):
    '''Just another crossentropy'''
    ly = K.sum(y_true*y_pred)
    ly2 = K.sum(y_pred*y_pred)
    ly3 = K.sum(y_true*y_true)
    out = 1.0-ly/K.sqrt(ly2*ly3)
    return out
#Pearson Correlation
def PCC(y_true,y_pred):
    '''Just another crossentropy'''
    lt1 = y_true-K.mean(y_true)
    lt2 = y_pred-K.mean(y_pred)
    
    ly = K.sum(lt1*lt2)
    ly2 = K.sum(lt1*lt1)
    ly3 = K.sum(lt2*lt2)
    out = ly/K.sqrt(ly2*ly3)
    
    return out

def PCCl(y_true,y_pred):
    '''Just another crossentropy'''
    lt1 = y_true-K.mean(y_true)
    lt2 = y_pred-K.mean(y_pred)
    
    ly = K.sum(lt1*lt2)
    ly2 = K.sum(lt1*lt1)
    ly3 = K.sum(lt2*lt2)
    out = ly/K.sqrt(ly2*ly3)
    
    return 1.0-out

opt = Adam(lr=0.05, beta_1=0.7, beta_2=0.999, decay=0.001)
#opt = SGD(lr=100, decay=1e-6, momentum=1.9)

model.compile(loss=PCCl, optimizer=opt, metrics=[MOC])

Wsave = model.get_weights()


ytr = 0.05*ymax*randn(Y.shape[0],Y.shape[1],Y.shape[2],1,1)
Y1 = Y +ytr
history1 = model.fit(X, Y1, batch_size = 100, epochs = 10)

ytr = 0.05*ymax*randn(Y.shape[0],Y.shape[1],Y.shape[2],1,1)
Y1 = Y +ytr
history2 = model.fit(X, Y1, batch_size = 100, epochs = 10)

ytr = 0.05*ymax*randn(Y.shape[0],Y.shape[1],Y.shape[2],1,1)
Y1 = Y +ytr
history3 = model.fit(X, Y1, batch_size = 100, epochs = 10)
