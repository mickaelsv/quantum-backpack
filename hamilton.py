#!/home/hllmnt/python_venv/bin/python3
import numpy as np

def hijij(v, vis,  uis, l, n):
    s = np.sum(vis)
    
    his = np.zeros(n)

    c = np.identity(2**n)

    c *= v**2 - 0.5*np.sum(uis) + l*(s**2)/4

    for i, (vi, ui) in enumerate(zip(vis, uis)):
        hi = -0.5*ui + l*(vi**2 + 0.5*vi*s - v*vi)
        his[i] = hi
        print("h{}= {}".format(i, hi))

    jijs = np.zeros((n, n))

    for j, vj in enumerate(vis):
        for i in range(j):
            jij = 2*l*vj*vis[i]
            jijs[i, j] = jij
    return his, jijs, c

def zi(i, n):
    ret = np.identity(2**n)
    ret[2**i] = -1
    return ret

def hc(his, jijs, c, n):
    hc = np.zeros((2**n, 2**n)) + c

    for i in range(n):
        hc += his[i]*zi(i, n)
        for j in range(i):
            hc += jijs[j, i]*zi(j, n)*zi(i, n)
    print(hc)

his, jijs, c = hijij(4, np.array([1, 2, 3]), np.array([1, 20, 1]), 100, 3)

hc(his, jijs, c, 3)


