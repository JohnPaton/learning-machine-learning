import random
import numpy as np
import matplotlib.pyplot as plt
import time
from subprocess import call
import os


# Polynomial with coefficients coefs
def poly(coefs,x):
    y = 0
    for i in range(0,len(coefs)):
        y += coefs[i]*(x**i)
    return y

# Cost function for individual given data
def cost(ind,data):
    c = 0
    
    for p in data:
        y = poly(ind,p[0])
        c += ((y-p[1])**2)#/((abs(y)+abs(p[1]))/2)

##    dataX = [i[0] for i in data];
##    dataY = [i[1] for i in data];
##
##    indY = [poly(ind,x) for x in dataX];
##    diffY = [dataY[i] - indY[i] for i in range(0,len(data))];
##    res2 = sum([r**2 for r in diffY]);
##
##    sigma = np.std(diffY);
##
##    c = res2
##    

    return c

# simulate N data points using polynomial coefficients coefs with domain xrange
# returns data as a list [[x1,y2],[x2,y2],...] sorted in increasing x order
def simData(xrange, N, coefs):
    data = []
    for i in range(0,N):
        x = random.uniform(-xrange,xrange)
        y = poly(coefs,x)
        data.append([x,y])

    y = [d[1] for d in data]
    # Some additive and some multiplicative noise
    for d in data:
        d[1] *= random.gauss(1, 0.1)
        d[1] += random.gauss(0, 0.1*(max(y) - min(y)))

    return sorted(data, key = lambda x: x[0])
        

# generate a child using two parents, with some chance of mutation
def breed(m,f):
    n = len(m)
    c = []
    for i in range(0,n):
        c.append(m[i] if random.randint(0,1) else f[i])
        if not random.randint(0,9):
            c[i] *= random.gauss(1,0.5)
        if not random.randint(0,9): # some additive mutation so we don't get stuck at zero
            c[i] += random.gauss(0,1)

    return c

# initialize a generation of popSize individuals, each having indSize coefficients
# returns a list of individuals which are lists of polynomial coefficients [c0,c1,...]
# optional: use data range to get a good range of initial individuals
def initGen(popSize,indSize,data=[]):
    currentGen = []
    ranges = []
    if not data:
        ranges = [10 for i in range(0,indSize)]
    else:
        datarange = max([p[1] for p in data]) - min([p[1] for p in data])
        for i in range(0,indSize):
            exp = 1 if not i else i

            ranges.append(datarange**(1/exp))
        
    for i in range(0,popSize):
        ind = []
        for j in range(0,indSize):
            r = ranges[j];
            ind.append(random.uniform(-r,r))
        currentGen.append(ind)
    return currentGen

# return new generation of individuals, with lower cost individuals having greater chance
# to breed
def gen(current,data):
    nBreeders = int(len(current)/2);
    genSorted = sorted(current, key = lambda x:cost(x,data))
    
    nextGen = [];
    # top 50% guaranteed at least two children
    for i in range(0,nBreeders):
        nextGen.append(breed(genSorted[i],genSorted[random.randint(0,i)]))
        # everyone has a chance to breed <3
        nextGen.append(breed(genSorted[i],genSorted[random.randint(0,nBreeders)]))

    return nextGen

# evolve a best degree deg polynomial fit to the data.
# Returns final generation sorted by cost (best fit first)
# optional: choose population size (default 30), fix number nGens of generations
#           (0 implies adaptive stopping once settled into a solution), provide target
#           coefficients for display purposes, live plot progress, make gif of progress.
#
def evolve(data,deg,popSize=30,nGens=0,target=[],plot=True,toGif=False):
    dataX = [p[0] for p in data]
    dataY = [p[1] for p in data]
    indstep = (max(dataX)*1.1 - min(dataX)*1.1)/100 # smooth plotting of individuals
    indX = np.arange(min(dataX)*1.1,max(dataX)*1.1+indstep,indstep)

    
    currentGen = initGen(popSize,deg,data)
    top10cost = [];

    if plot or toGif:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    # Delete old frames. BE CAREFUL WITH THIS
    cwd = os.getcwd()
    figdir = cwd+r'\figs'
    if os.path.exists(figdir) and toGif:
        cmd = 'del "'+figdir+r'\gen*.png"'
        call(cmd,shell=True)
    elif toGif:
        os.makedirs(figdir)

    stop = False
    i = 0
    rollingAvg = 20 # consider last 20 gens for adaptive stopping
    stopLevel = 0.0005 # stop when var/mean of most recent top10cost is < stopLevel
    
    targstr = ', '.join([str(t) for t in target]) if target else 'unknown'
    
    while not stop:
        
        genSorted = sorted(currentGen, key = lambda x:abs(cost(x,data)))
        top10cost.append(sum([cost(ind,data) for ind in genSorted[0:int(popSize/10)]]))
        
        if nGens:
            if i+1 == nGens:
                stop = True
                totalGens = nGens;
        # adaptive stopping condition
        elif len(top10cost) > rollingAvg:
            m = np.mean(top10cost[i-rollingAvg:])
            v = np.var(top10cost[i-rollingAvg:])
            
            if plot:
                print(v/m)

            if v/m < stopLevel:
                stop = True
                totalGens = i
                
        if not stop:
            # plot current generation
            if plot or toGif:
                ax.clear()
                ax.axhline(y=0, color='k')
                ax.axvline(x=0, color='k')
                ax.scatter(dataX,dataY,color = 'k')
                ax.set_xlim(min(indX),max(indX))
                ax.set_ylim(min(0,min(dataY)*2),max(0,max(dataY)*2))
                beststr = ', '.join(['{:.2f}'.format(c) for c in genSorted[0]])
                t = 'Population Size: {}, Target Coefs: {}'.format(popSize,targstr)
                t += '\nCurrent Gen: {}, Gen Best: {}'.format(i+1,beststr)
                plt.title(t)
                #plt.draw()
                #plt.pause(0.01/(i+1))

                for j in reversed(range(0,popSize)):
                    c = 'g' if j<popSize/2 else 'r'
                    indY = [poly(genSorted[j],x) for x in indX]
                    ax.plot(indX,indY,color=c)
                if plot:
                    plt.draw()
                if toGif:
                    for j in range(0,int(20/(i+1))+1):
                        f = './figs/gen'+'{:04d}'.format(i+1)
                        f += '('+str(j)+')' if j else ''
                        plt.savefig(f)

        #print('Best:',''.join(top),'\t\tTotal Cost:',totcost)
                if plot:
                    plt.pause(2.0/(i+1))

            # breed next generation
            nextGen = gen(currentGen,data)
            currentGen = nextGen
            i += 1;

    # show final result
    if plot or toGif:
        ax.clear()
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.scatter(dataX,dataY,color = 'k')
        ax.set_xlim(min(indX),max(indX))
        ax.set_ylim(min(0,min(dataY)*2),max(0,max(dataY)*2))
        guess = genSorted[0]
        guessstr = ''
        targstr = ''
        for j in range(0,len(guess)):
            if j == 0:
                guessstr += 'y = {:.2f}'.format(guess[j])
                targstr += 'y = {:}'.format(target[j]) if target else 'unknown'
            elif j==1:
                guessstr += ' + {:.2f} x'.format(guess[j])
                targstr += ' + {:} x'.format(target[j]) if target else ''
            else:
                guessstr += ' + {:.2f} x^{}'.format(guess[j],j)
                targstr += ' + {:} x^{}'.format(target[j],j) if target else ''
        
        t = 'Target was: {}'.format(targstr)
        t += '\nBest after {} gens: {}'.format(str(i),guessstr)
        plt.title(t)
        indY = [poly(genSorted[0],x) for x in indX]
        ax.plot(indX,indY,color='g',linewidth=3)
        if toGif:
            for j in range(0,40):
                f = './figs/gen{:04d}finalguess'.format(i)
                f += '('+str(j)+')' if j else ''
                plt.savefig(f)
        if plot:
            plt.show()

        # create gif using Image Magick (slow)
        if toGif:
            print('Stopped after {} generations'.format(totalGens))
            print('Creating gif (may take some time)...')
            pngs = figdir + r'\gen*.png'
            gif = figdir + r'\{}.gif'.format(int(time.time()))
            cmd = r'magick convert -delay 10 "' + pngs + r'" "' + gif + r'"'
            try:
                call(cmd, shell=True)
                print('Gif complete')
                print('Saved to',gif)
            except Exception as e:
                print(str(e))


    s = 'Stopped after {} generations'.format(totalGens)
    if not plot and not toGif:
        print(s)
        
    return sorted(currentGen, key = lambda x:abs(cost(x,data)))


# Check out how it all works
popSize = 50
nGens = 200
target = [0,-50,0,1]

data = simData(10,200,target)
dataX = [p[0] for p in data]
dataY = [p[1] for p in data]

g = evolve(data,len(target),popSize,plot=True,toGif=False,target=target)
best = g[0]
npbest = list(reversed( np.polyfit(dataX,dataY,len(target)-1)))

err = [(target[i] - best[i]) for i in range(0,len(target))]
nperr  = [(target[i] - npbest[i]) for i in range(0,len(target))]

print('Gen:',', '.join(['{:.2f}'.format(x) for x in err]))
print('Npy:',', '.join(['{:.2f}'.format(x) for x in nperr]))
print()


    
    
        

    
        
