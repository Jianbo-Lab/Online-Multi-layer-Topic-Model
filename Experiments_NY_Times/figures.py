import numpy as np
import matplotlib
from pylab import *
import matplotlib.pyplot  as pyplot

# Learning rates Experiments.
def read_in_results(K1,K2,rho1,rho2):
    with open('results for learning parameters/perplexity_%d_%d_%d_%f.txt'%(K1,K2,rho1,rho2),'r') as f:
        a = np.array(f.readlines())
    # Compensate for the subsampling of the population of documents (Blei, 2010).
    aa = 10
    return a.astype(np.float) * aa

rho1_seq = [0,256,1024]
rho2_seq = [0.55,0.6,0.7,0.8,0.9,1]
learning_para_choices0 = [read_in_results(100,3,0,a) for a in rho2_seq]
learning_para_choices1 = [read_in_results(100,3,256,a) for a in rho2_seq]
learning_para_choices2 = [read_in_results(100,3,1024,a) for a in rho2_seq]


iterations = np.array(range(65)+[15 * i for i in range(5,310)])
numdoc = np.multiply(np.add(iterations,1),64)


fig = pyplot.figure()
ax = fig.add_subplot(1,1,1)
linestyles = ['-', '-.', '--', ':','1','2','d','p']
ax.plot(numdoc,learning_para_choices1[0], color='black',label = '$\kappa$ = 0.55', lw=2)
ax.plot(numdoc,learning_para_choices1[1],color='blue',label = '$\kappa$ = 0.6', lw=2)
ax.plot(numdoc,learning_para_choices1[2], color='red',label = '$\kappa$ = 0.7', lw=2)
ax.plot(numdoc,learning_para_choices1[3], color='green',label = '$\kappa$ = 0.8', lw=2)
ax.plot(numdoc,learning_para_choices1[4], color='purple',label = '$\kappa$ = 0.9', lw=2)
ax.plot(numdoc,learning_para_choices1[5], color='yellow',label = '$\kappa$ = 1', lw=2)
 
ax.set_xscale('log')
ax.set_yscale('log')
plt.plot(10**1.5,10**5, 'ko', marker=r'$\uparrow$', markersize=20)
plt.plot(10**1.6,10**4.5, 'ko', marker=r'$\kappa$ increasing', markersize=60)
ax.legend(bbox_to_anchor=(0.60, 0.95), loc=2, borderaxespad=0.)
xlabel("Number of documents")
ylabel("Perplexity")
title("Increasing forgetting rate kappa with other parameters fixed")  

show()
fig.savefig('kappa.eps', format='eps', dpi=1000)

fig = pyplot.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(numdoc,learning_para_choices0[0],linestyles[1], color='black',label = 'tau = 0', lw=2)
ax.plot(numdoc,learning_para_choices1[0],linestyles[0], color='blue',label = 'tau = 256', lw=2)
ax.plot(numdoc,learning_para_choices2[0],linestyles[2], color='red',label = 'tau = 1024', lw=2) 
 
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(bbox_to_anchor=(0.65, 0.95), loc=2, borderaxespad=0.)
xlabel("Number of documents")
ylabel("Perplexity")
plt.plot(10**2.4,10**5.5, 'ko', marker=r'$\uparrow$', markersize=20)
plt.plot(10**1.8,10**5.5, 'ko', marker=r'$\tau$ increasing', markersize=70)
plt.plot(10**1.5,10**5.2, 'ko', marker=r'$\tau = 1024$', markersize=25)
plt.plot(10**1.5,10**5, 'ko', marker=r'$\tau = 256$', markersize=30)
plt.plot(10**1.5,10**4.5, 'ko', marker=r'$\tau = 0$', markersize=30)
title("Vary delay with other parameters fixed") 
#ax.set_yscale('log')
show()
fig.savefig('tau.eps', format='eps', dpi=1000)

# Experiment 2: Selection of K2.
def read_in_results(K1,K2,rho1,rho2):
    with open('results for small K2/perplexity_%d_%d_%d_%f.txt'%(K1,K2,rho1,rho2),'r') as f:
        a = np.array(f.readlines())
        # Compensate for the subsampling of the population of documents (Blei, 2010).
        aa = 10
    return a.astype(np.float) * aa
K1 = 100
K2_seq = [1,7,10,30]
rho1 = 1024
rho2 = [0.7,1]
# Create files and python jobs so as to run my codes on Linux Cluster.
 
K2_choices100 = [read_in_results(100,a,256,0.7) for a in K2_seq]

fig = pyplot.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(numdoc,K2_choices100[0],linestyles[1], color='black',label = 'K2 = 1 (Online LDA)', lw=2)

ax.plot(numdoc,K2_choices100[1],linestyles[0], color='blue',label = 'K2 = 7', lw=2)

ax.plot(numdoc,K2_choices100[3],linestyles[2], color='green',label = 'K2 = 30', lw=2)
 
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(bbox_to_anchor=(0.45, 0.95), loc=2, borderaxespad=0.)
xlabel("Number of documents")
ylabel("Perplexity")
plt.plot(10**1.5,10**5, 'ko', marker=r'$K_2 = 1$', markersize=30)
plt.plot(10**1.5,10**5.2, 'ko', marker=r'$K_2 = 7$', markersize=30)
plt.plot(10**1.5,10**5.45, 'ko', marker=r'$K_2 = 30$', markersize=30)
plt.plot(10**5.5,10**2.4, 'ko', marker=r'$\uparrow K_2 = 7$', markersize=30)
title("Vary K2 with other parameters fixed") 
#ax.set_yscale('log')
show()
fig.savefig('K2.eps', format='eps', dpi=1000)

# Experiment 3: Selection of K1.
def read_in_results(K1,K2,rho1,rho2):
    with open('results for small K2/perplexity_%d_%d_%d_%f.txt'%(K1,K2,rho1,rho2),'r') as f:
        a = np.array(f.readlines())
        # Compensate for the subsampling of the population of documents (Blei, 2010).
        aa = 10
    return a.astype(np.float) * aa
K1_seq = [10,30,50,100]
K2 = 10
rho1 = 256
rho2 = 0.7
# Create files and python jobs so as to run my codes on Linux Cluster.
K1_choices = [read_in_results(a,10,256,0.7) for a in K1_seq]
#K2_choices150 = [read_in_results(150,a,256,0.7) for a in K2_seq]

fig = pyplot.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(numdoc,K1_choices[0],linestyles[3], color='black',label = 'K1 = 10', lw=2)

ax.plot(numdoc,K1_choices[1],linestyles[1], color='blue',label = 'K1 = 30', lw=2)

ax.plot(numdoc,K1_choices[2],linestyles[2], color='red',label = 'K1 = 50', lw=2)

ax.plot(numdoc,K1_choices[3],linestyles[0], color='purple',label = 'K1 = 100', lw=2)
 
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(bbox_to_anchor=(0.65, 0.95), loc=2, borderaxespad=0.)
xlabel("Number of documents")
ylabel("Perplexity")
plt.plot(10**1.5,10**5, 'ko', marker=r'$\uparrow$', markersize=20)
plt.plot(10**1.5,10**4.5, 'ko', marker=r'$K_1$ increasing', markersize=70)
plt.plot(10**5.7,10**2.7, 'ko', marker=r'$\uparrow$', markersize=20)
plt.plot(10**5.3,10**2.3, 'ko', marker=r'$K_1$ decreasing', markersize=70)
title("Vary K1 with other parameters fixed") 
#ax.set_yscale('log')
show()
fig.savefig('K1.eps', format='eps', dpi=1000)