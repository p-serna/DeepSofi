from numpy import *
from matplotlib.pylab import *

def s0(t,nu,phi=-1e-6): return(sign(sin(2.0*pi*nu*t+phi)))
def si(t,th=1.0,mu = 0,s = 1,dt=1e-2,x0=0):
    if isscalar(t):
        ts = arange(0,t,dt)
    else:
        ts = t
    W = s*sqrt(dt)*randn(len(ts))
    x = W*0
    x[0] = x0
    for i,ti in enumerate(ts[:-1]):
        x[i+1] = x[i]+th*(mu-x[i])*dt+W[i]
    return x
def s1(t,s0=0,l1=1.0,l2 = None,t0=None):
    if l2 is None:
        l2 = l1
    if abs(s0)>1e-6 and abs(s0-1)>1e-6:
        print("Sorry, s1 will take values 0 or 1, choose either as a initial state")
        return
    tf = max(t)
    xt = -log(rand(int(1.0*tf*l1)))/l1
    yt = -log(rand(int(1.0*tf*l2)))/l2
    if t0 is None:
        if s0==0:
            tchange = array([[t1,t2] for t1,t2 in zip(xt,yt)]).flatten()
        else:
            tchange = array([[t1,t2] for t1,t2 in zip(yt,xt)]).flatten()
            
    tchange = cumsum(tchange)
    tchange = tchange[tchange<=tf]
    
    s1 = t*0
    ta = 0
    state = s0
    for tc in tchange:
        sel = (t>=ta)*(t<tc)
        if sum(sel)==0:
            # Hay un problema!
            pass
        s1[sel] = state
        state = 1-state
        ta = tc
    return s1

def cluster(t,nt=5):
    stf = []
    s1tf = t*0+1
    pars  = {"delta":[],"std":[],"on":[],"ls":[],"Veff":[]}
    for i in range(nt):
        delta = 1.2+1.0*rand()
        pars["delta"].append(delta)
        std0 = randn()*0.02+0.1
        std1 = randn()*0.05+0.3
        si1 = si(t,th=1.0,mu=1.0,s=std0,dt=1000e-3)
        si2 = si(t,th=1.0,mu=delta,s=std1,dt=1000e-3)
        pars["std"].append([std0,std1])
        on = rand()>0.3
        pars["on"].append(on)
        
        l1o = rand()*10
        l2o = rand()*10
        pars["ls"].append([l1o,l2o]) 
        s1t = s1(t,s0=0,l1=l1o/1000,l2=l2o/1000)
        if not on:
            s1t[8000:] = 1
        # ~ st = -s0(t,nu)*0.05*(1-s1t)/2.0+ si1*s1t+si2*(1-s1t)
        st = si1*s1t+si2*(1-s1t)
        pars["Veff"].append(0.05)
        s1tf = s1t*s1tf
        
        stf.append(st)
    stf = array(stf)
    stf = sum(stf,axis=0)
    return((stf,s1tf,pars))

def main():
	dset = []
	t = arange(0,5*14000,5)
	s0 = sign(sin(2*pi*t/20+1e-4))
	save("syndat/s0.npy",s0[2000:])
	for i in range(4200):
		if rand()<0.7:
			nt = 1
			st,s1t,pars = cluster(t,nt)
		else:
			nt = randint(2,13)
			st,s1t,pars = cluster(t,nt)
			
		st = st/mean(st)
		save("syndat/d"+str(i).zfill(4)+".npy",st[2000:])
		save("syndat/r"+str(i).zfill(4)+".npy",s1t[2000:])
		
		if nt > 1:
			delta = mean(array(pars["delta"]))
			l1o,l2o = mean(array(pars["ls"]),axis=0)
		else:
			delta = pars["delta"][0]
			l1o,l2o = pars["ls"][0]
		dset.append([delta,l1o,l2o,nt])


	save("syndat/pars.npy",dset)

if __name__ == "__main__":
    main()


