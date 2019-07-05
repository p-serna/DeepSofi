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
def s1(t,s0=0,l1=1.0,l2 = None,t0=0.0):
    if l2 is None:
        l2 = l1
    if abs(s0)>1e-6 and abs(s0-1)>1e-6:
        print("Sorry, s1 will take values 0 or 1, choose either as a initial state")
        return
    tf = max(t)
    xt = -log(rand(int(1.0*tf*l1)))/l1
    yt = -log(rand(int(1.0*tf*l2)))/l2
    if s0==0:
        tchange = array([[t1,t2] for t1,t2 in zip(xt,yt)]).flatten()
    else:
        tchange = array([[t1,t2] for t1,t2 in zip(yt,xt)]).flatten()
        
    tchange = t0+cumsum(tchange)
    tchange = tchange[tchange<=tf]
    
    s1 = t*0
    ta = 0
    state = s0
    selfn = 0
    selin = 0
    indcs = arange(len(t))
    for tc in tchange:
        selfn = indcs[(t<tc)][-1]
        if selfn<=selin:
            # Hay un problema! No se como resolverlo por el momento...
            selfn = selfn+1
        s1[selin:selfn] = state
        state = 1-state
        selin = selfn 
    return s1

def cluster(t,nt=5,par0 = None):
    stf = []
    s1tf = t*0+1
    lent = len(t)
    tf = t[-1]
    if par0 is None:
        par0 = {"Veff":[]}
    if "delta" not in par0.keys():
        par0["delta"]=1.2+rand()*1.0
    if "std" not in par0.keys():
        par0["std"] = [0.02*randn()+0.1,0.05*randn()+0.3]
    if "on" not in par0.keys():
        par0["on"] = [0.3,0.2]
    if "ls" not in par0.keys():
        par0["ls"] = [10*rand(),10*rand()]
        
    pars  = {"delta":[],"std":[],"on":[],"ls":[],"Veff":[]}
    for i in range(nt):
        delta = par0["delta"] 
        #[0]+par0["delta"][1]*rand()
        pars["delta"].append(delta)
        std0,std1 = par0["std"]
        si1 = si(t,th=1.0,mu=1.0,s=std0,dt=1)
        si2 = si(t,th=1.0,mu=delta,s=std1,dt=1)
        pars["std"].append([std0,std1])
        on = par0["on"]
        s00 = t*0
        s00[int(lent*on[0]):int(lent*(on[1]+on[0]))] = 1
        pars["on"].append(par0["on"])
        
        l1o = par0["ls"][1]   #on
        l2o = par0["ls"][0]   #off
        pars["ls"].append([l1o,l2o]) 
        s1t = s1(t,s0=0,l1=l1o,l2=l2o)
        s1t[int(lent*0.8):] = 1
        # ~ st = -s0(t,nu)*0.05*(1-s1t)/2.0+ si1*s1t+si2*(1-s1t)
        s1t = 1-(1-s1t)*s00
        st = si1*s1t+si2*(1-s1t)
        #pars["Veff"].append(0.05)
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


