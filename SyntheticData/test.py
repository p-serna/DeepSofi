
%pylab


T = 40
nbs = 100000
a = randn(10,T)

boots = zeros((10,nbs))
for i in range(10):
    select = randint(0,T,(nbs,T))
    boots[i,:]= mean(a[i,select],axis=-1)

from scipy.stats import ttest_1samp

m = mean(a,axis=1)
s = std(a,axis=1)/sqrt(T)
pe = []; p = []
for i in range(10):
    pe.append(sum(abs(boots[i,:]-0.0)<s[i])/nbs)
    
    p.append(ttest_1samp(a[i,:],0.0).pvalue)
pe
p

a2 = randn(100,T)
m2 = mean(a2,axis=1)
