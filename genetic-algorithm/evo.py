import random

def cost(ind,target):
    n = len(target)
    c = 0;
    for i in range(0,n):
        c += (ind[i] - target[i])**2

    return c;


def breed(m,f):
    n = len(m);
    c = [];
    mut = [-1,1];
    nMut= len(mut)
    for i in range(0,n):
        c.append(m[i] if random.randint(0,1) else f[i])
        if not random.randint(0,20):
            c[i] = (c[i] + mut[random.randint(0,nMut-1)])%10;


    return c;

def initGen():
    currentGen = [];
    for i in range(0,popSize):
        ind = [];
        for j in range(0,len(target)):
            ind.append(random.randint(0,9));
        currentGen.append(ind);
    return currentGen;

def gen(current):
    nBreeders = int(len(current)/2);       
    genSorted = sorted(current, key = lambda x:cost(x,target))

    nextGen = [];
    for i in range(0,nBreeders):
        nextGen.append(breed(genSorted[i],genSorted[random.randint(0,nBreeders)]))
        nextGen.append(breed(genSorted[i],genSorted[random.randint(0,nBreeders)]))

    return nextGen;

def evolve(popSize,nGens,target):
    currentGen = initGen();

    for i in range(0,nGens):
        totcost = sum([cost(i,target) for i in currentGen])
        top = sorted(currentGen, key = lambda x:cost(x,target))[0];
        top = [str(i) for i in top];

        
        print('Best:',''.join(top),'\t\tTotal Cost:',totcost)
        nextGen = gen(currentGen);
        currentGen = nextGen;
        
    return sorted(currentGen, key = lambda x:cost(x,target));


popSize = 10;
nGens = 50;

#target = [3,1,4,1,5,9,2,6,5,3,5,9]
target = [0,1,2,3,4,5,6,7,8,9]
g = evolve(popSize,nGens,target)




    
    
        

    
        
