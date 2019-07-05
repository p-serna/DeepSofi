from generating_lib2 import *
from scipy.integrate import dblquad
from scipy.special import erf
from numpy import *
from matplotlib.pylab import *


def generatepos(nt,w=1,h=1,extraw=0.2,size=[0.1,0.2]):
    xs = (extraw+fwidth)*rand(npt)-extraw/2.0
    ys = (extraw+fheight)*rand(npt)-extraw/2.0
    ws = exp(randn(npt)*size[0])*size[1]
    return((xs,ys,ws))

def gausint(x,y,w=1.0,x0=0.0,y0=0.0):
    intx = (1.0 + erf((x-x0+0.5)/sqrt(2)/w))/2.0
    intx = intx-(1.0 + erf((x-x0-0.5)/sqrt(2)/w))/2.0
    inty =    (1.0 + erf((y-y0+0.5)/sqrt(2)/w))/2.0
    inty = inty-(1.0 + erf((y-y0-0.5)/sqrt(2)/w))/2.0
    return(intx*inty)
    
def gaus2d(x,y,w=1.0,x0=0.0,y0=0.0):
    return(exp(-((x0-x)**2+(y0-y)**2)/2.0/(w)**2)/(2*pi*w**2))

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
                xi2 = xi/magn
                yi2 = yi/magn
                integralt2 = gausint(xi2,yi2,w,x,y)
                orimg[yi,xi,i] = integralt2
    return(sum(orimg,axis=-1))

def factorimg(x,ys = None,ws = None, resolution = None,wm=1.0, magn=1):
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
    fftim  = fftshift(fft2(img))
    fftim  =  pad(fftim,(sh2[0],sh2[1]),'constant')    
    imgn = real(ifft2(ifftshift(fftim)))
    return(imgn)

def ftvaugment(img,magn=2):
    sh = img.shape
    sh2 = (sh*(magn-1))
    sh2 = ((sh2[0])//2,(sh2[0])//2)
    fftim  = fftshift(fft2(img,axes=(0,1)),axes=(0,1))
    fftim  =  pad(fftim,(sh2[0],sh2[1]),'constant')    
    imgn = real(ifft2(ifftshift(fftim,axes=(0,1)),axes=(0,1)))
    imgn = imgn[:,:,sh2[0]:-sh2[0]]
    return(imgn)

def addingblinking(factor,t,bg= 1000.0, sigma= 400.0,npt=None):
    fwidth,fheight,npt = factor.shape
    img = randn(fwidth,fheight,len(t))*sigma+bg
    for i in range(npt):
        t0 = rand()
        wt = (1.0-t0)*rand()
        delta = 1.2+rand()*1.0
        par0={"ls":[0.02,0.20],"delta":delta,"on":[t0,wt]}
        st,s1t,pars = cluster(t,1,par0=par0)
        st = st*bg
        img = img + reshape(st,(1,1,len(t)))*reshape(factor[:,:,i],(fwidth,fheight,1))
    return(img)







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
  
# ~ def main():
t = arange(0,5000)
npt = 10

fwidth = 10
fheight = 10
extraw = 1

wm = 2.0
#xs,ys,ws = generatepos(npt,fwidth,fheight,extraw,[0.1,.50])
#orimg = originalimg(xs,ys,ws,(10,10),magn=1); figure(); imshow(log(orimg))
#factor = factorimg(xs,ys,ws,(10,10),wm=1.0,magn=1); figure(); imshow(log(sum(factor,axis=-1)))
#st,_,_ = cluster(t,1,par0 = {"ls":[10,0.1]})
#video = addingblinking(factor,t)

dset = []
t = arange(0,5000,1)
print("Starting..........")
for i in range(900):
    if i%300==0:
        print("generated",i,"samples")
    npt = randint(10)
    fwidth = 10
    fheight = 10
    extraw = 2
    wm = 4+2*rand()
    xs,ys,ws = generatepos(npt,fwidth,fheight,extraw,[0.1,0.2])
    
    #orimg = originalimg(xs,ys,ws,(10,10),magn=2)
    factor = factorimg(xs,ys,ws,(10,10),wm=wm,magn=1) 
    video = addingblinking(factor,t)
        
    video = video/mean(video.flatten())
    save("syndat/d"+str(i).zfill(4)+".npy",video[:,:,1000:])
    orimg = originalimg(xs,ys,ws,(10,10),magn=2)

    save("syndat/o2_"+str(i).zfill(4)+".npy",orimg)
    #orimg = originalimg(xs,ys,ws,(10,10),magn=4)
    #save("syndat/o4_"+str(i).zfill(4)+".npy",orimg)
    #orimg = originalimg(xs,ys,ws,(10,10),magn=8)
    #save("syndat/o8_"+str(i).zfill(4)+".npy",orimg)
    #save("syndat/p"+str(i).zfill(4)+".npy",column_stack((xs,ys,ws)))
