def Layers(X,layers = None):
    if layers is None:
        return(X)
    for key,layer in layers.items():
        X = layer(X)
    return(X)


def LConv3D(nconv,kernel,strides = (1,1,1),padding="same",dropout=0.3,activation= "relu"):
    def multilayer(X):
        X = Dropout(dropout)(X)
        X = Conv3D(nconv,kernel,strides = strides,padding=padding)(X)
        X = Activation(activation)(X)
        X = BatchNormalization()(X)
        return(X)
    return(multilayer)
    
def LMaxPool(kernel,strides = (1,1,1),padding="same"):
    def multilayer(X):
        X = MaxPooling3D(kernel,strides,padding=padding)(X)
        return(X)
    return(multilayer)

def LAvgPool(kernel,strides = (1,1,1),padding="same"):
    def multilayer(X):
        X = AveragePooling3D(kernel,strides,padding=padding)(X)
        return(X)
    return(multilayer)
    
def LDense(units,dropout=0.3,activation= "relu"):
    def multilayer(X):
        X = Dropout(dropout)(X)
        X = Dense(units,activation=activation)(X)
        X = BatchNormalization()(X)
        return(X)
    return(multilayer)
    
    
layerd = {"Conv": LConv3D, "MaxPool": LMaxPool, "AvgPool": LAvgPool,
"Dense" : LDense}
#, "Dropout": Dropout, "Activation": Activation, "normalization": BatchNormalization

keys = layerd.keys()

layers = {}
layers[0] = layerd[keys[0]](80,(4,4,8),strides=(1,1,3),padding="same",dropout=0.3,activation="relu")
layers[1] = layerd[keys[2]]((4,4,20),strides=(1,1,3),padding="same")
layers[2] = layerd[keys[0]](40,(1,1,12),strides=(1,1,12),padding="valid",dropout=0.3,activation="relu")
layers[3] = layerd[keys[3]](20,activation="relu")
layers[4] = layerd[keys[3]](5,activation="relu")


n_layers = 1
npar = 5
for ilayer in range(n_layers):
    ir = randint(4,size = npar)
    parlay = []
    for ipar in range(npar):
        parlay.append(layerd[keys[ir]]())

layerp = dict(layers)

layers[0] = layerd[keys[0]](80,(4,4,8),strides=(1,1,3),padding="same",dropout=0.3,activation="relu")
layers[1] = reshape((20,20,1,80*34))


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
    
    Xa = MaxPooling3D((2,2,40),strides=(1,1,5),padding="same")(X)
    Xb = AveragePooling3D((2,2,40),strides=(1,1,5),padding="same")(X)
    Xa2 =  Lambda(lambda x: x * x)(Xa)
    Xb2 = Lambda(lambda x: x * x)(Xb)
    Xa3 =  Lambda(lambda x: x **3)(Xa)
    Xb3 = Lambda(lambda x: x **3)(Xb)
    Xa4 =  Lambda(lambda x: x **4)(Xa)
    Xb4 = Lambda(lambda x: x **4)(Xb)

    X = Concatenate()([Xa,Xb,Xa2,Xb2,Xa3,Xb3,Xa4,Xb4])
    X = BatchNormalization()(X)   
    
    X = Layers(X)    
    
    X = Dropout(0.3)(X)
    X = Dense(1,activation="relu")(X)

    # Defining the model
    
    model = Model(inputs = X_input, outputs = X)
    
    return model
