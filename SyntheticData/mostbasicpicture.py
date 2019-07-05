from generating_lib import *
t = arange(0,5*14000,5)

st,s1t,pars = cluster(t,1)

plot(t,st)

 
npt = 10

fwidth = 10
fheight = 10
extraw = 2
xs = (extraw+fwidth)*rand(npt)-extraw/2.0
ys = (extraw+fheight)*rand(npt)-extraw/2.0
ws = exp(randn(npt)*.1)*.2
wm = 5

factor = zeros((fwidth,fheight,npt))
 
for xi in range(fwidth):
    for yi in range(fheight):       
        for i in range(npt):
            x,y,w = (xs[i],ys[i],ws[i])
            factor[xi,yi,i] = exp(-((xi-x)**2+(yi-y)**2)/2.0/(w*wm)**2) #/sqrt(2*pi*w**2)

img = zeros((fwidth,fheight,len(t)))
img = randn(fwidth,fheight,len(t))*1000+5000

for i in range(npt):
    st,s1t,pars = cluster(t,1)
    st = st*5000
    img = img + reshape(st,(1,1,len(t)))*reshape(factor[:,:,i],(fwidth,fheight,1))
    
from scipy.special import comb

def cumulantWikit0(im,n=5,full=False) :
    if n == 1: return(mean(im,axis=-1))
    if n>=2:
        cun = mean(im**n,axis=-1)
        mui = {1:mean(im,axis=-1)}
        cui = {1:mui[1]}
        imt = array(im)
        for i in range(2,n+1):
            imt = imt*imt
            mui[i] = mean(imt,axis=-1)
            cui[i] = array(mui[i])
            for j in range(1,i):
                cui[i] += comb(i-1,j)*mui[j]*cui[i-j]
    if full:
        return(cui)
    else:
        return(cui[n])
            
cuis = cumulantWikit0(img,6,full=True)
figure(); imshow(cuis[2])
figure(); imshow(cuis[6]/cuis[2]**(6.0/2.0))

imshow(sum(factor,axis=-1))

from scipy.integrate import dblquad
from scipy.special import erf

def gausint(x,y,w=1.0,x0=0.0,y0=0.0):
    intx = (1.0 + erf((x-x0+0.5)/sqrt(2)/w))/2.0
    intx = intx-(1.0 + erf((x-x0-0.5)/sqrt(2)/w))/2.0
    inty =    (1.0 + erf((y-y0+0.5)/sqrt(2)/w))/2.0
    inty = inty-(1.0 + erf((y-y0-0.5)/sqrt(2)/w))/2.0
    return(intx*inty)
    
def gaus2d(x,y,w=1.0,x0=0.0,y0=0.0):
    return(exp(-((x0-x)**2+(y0-y)**2)/2.0/(w)**2)/(2*pi*w**2))

magn =4

orimg = zeros((magn*fwidth,magn*fheight,npt))
orimg2 = zeros((magn*fwidth,magn*fheight,npt))

for xi in range(magn*fwidth):
    for yi in range(magn*fheight):       
        for i in range(npt):
            x,y,w = (xs[i],ys[i],ws[i])
            x = x*magn
            y = y*magn
            #integralt,_ = dblquad(gaus2d,xi-0.5,xi+0.5,gfun=lambda x: yi-0.5,hfun=lambda x: yi+0.5,\
            #    args = (w,x,y))
            integralt2 = gausint(xi,yi,w,x,y)
            #print(integralt,integralt2)
            orimg[yi,xi,i] = integralt2
            #orimg2[xi,yi,i] = integralt

factor = zeros((fwidth,fheight,npt))
 
for xi in range(fwidth):
    for yi in range(fheight):       
        for i in range(npt):
            x,y,w = (xs[i],ys[i],ws[i])
            integralt2 = gausint(xi,yi,w*wm,x,y)
            factor[yi,xi,i] = integralt2 

figure()
imshow(sum(orimg,axis=-1))
figure()
imshow(sum(factor,axis=-1))

def originalimg(x,ys = None,ws = None, resolution = None, magn=1):
    xsh = x.shape
    xs = x
    if ys is None:
        if len(xsh)<3:
            print("Data do not have proper format")
            return
        else:
            ys = x[:,1]
            ws = x[:,2]
            xs = x[:,0]
    if ws is None:
        ws = ones(xsh[0])

    try:
        fwidth,fheight = resolution
    except:
        print("Resolution not specified correctly")
        xs = xs +min(xs)
        ys = ys +min(ys)
        fwidth = int(max(xs))+2
        fheight = int(max(ys))+2
        print("Resolution used:",fwidth,"x",fheight)

    orimg = zeros((magn*fwidth,magn*fheight,npt))
    for xi in range(magn*fwidth):
        for yi in range(magn*fheight):       
            for i in range(npt):
                x,y,w = (xs[i],ys[i],ws[i])
                x = x*magn
                y = y*magn
                integralt2 = gausint(xi,yi,w,x,y)
                orimg[yi,xi,i] = integralt2
    return(sum(orimg,axis=-1))

def factorimg(x,ys = None,ws = None,wm=1.0, resolution = None, magn=1):
    xsh = x.shape
    xs = x
    if ys is None:
        if len(xsh)<3:
            print("Data do not have proper format")
            return
        else:
            ys = x[:,1]
            ws = x[:,2]
            xs = x[:,0]
    if ws is None:
        ws = ones(xsh[0])

    try:
        fwidth,fheight = resolution
    except:
        print("Resolution not specified correctly")
        xs = xs +min(xs)
        ys = ys +min(ys)
        fwidth = int(max(xs))+2
        fheight = int(max(ys))+2
        print("Resolution used:",fwidth,"x",fheight)

    factor = zeros((magn*fwidth,magn*fheight,npt))
    for xi in range(magn*fwidth):
        for yi in range(magn*fheight):       
            for i in range(npt):
                x,y,w = (xs[i],ys[i],ws[i])
                x = x*magn
                y = y*magn
                integralt2 = gausint(xi,yi,w*wm,x,y)
                factor[yi,xi,i] = integralt2
    return(factor)

from scipy.fftpack import fft2,fftshift,ifftshift,ifft2

def ftaugment(img,magn=2):
    sh = img.shape
    sh2 = (sh*(magn-1))
    sh2 = ((sh2[0])//2,(sh2[1])//2)
    imgn =  fft2(img)
    fftim  = fftshift(fft2(imgn))
    fftim  =  pad(fftim,(sh2[0],sh2[1]),'constant')    
    imgn = real(ifft2(ifftshift(fftim)))
    return(imgn)
    
