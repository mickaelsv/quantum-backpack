#!/home/hllmnt/python_venv/bin/python3
import numpy as np

def hijij(v, vis,  uis, l, n):
    #Calcul des hi, Jij et c
    sv = np.sum(vis)
    sv2 = np.sum(vis**2)
    su = np.sum(uis)
    
    his = np.zeros(n)

    c = np.identity(2**n)

    c *= -0.5*su + l*(sv2 / 4 + sv**2 / 4 - v*sv + v**2)

    for i, (vi, ui) in enumerate(zip(vis, uis)):
        hi = -0.5*ui + l*(0.5*sv - v)*vi
        his[i] = hi
        print("h{}= {}".format(i, hi))

    jijs = np.zeros((n, n))

    for j, vj in enumerate(vis):
        for i in range(j):
            jij = l*vj*vis[i]/2
            jijs[i, j] = jij
            print("J{}{}= {}".format(i, j, jij))
    return his, jijs, c

def zi(i, n):
    #Calcul de l'opérateur Z_i
    ret1 = np.identity(2**i)
    ret2 = np.array([[1, 0], [0, -1]])
    ret3 = np.identity(2**(n-i-1))
    return np.kron(np.kron(ret1, ret2), ret3)

def hc(his, jijs, c, n):
    #Calcul de l'Hamiltonien
    hc = np.zeros((2**n, 2**n)) + c

    for i in range(n):
        hc += his[i]*zi(i, n)
        for j in range(i):
            hc += jijs[j, i]*zi(j, n)@zi(i, n)

    #print(hc)

    return hc

his, jijs, c = hijij(33, np.array([10, 12, 11, 30]), np.array([11, 21, 10, 100]), 1, 4)

# Avec cet exemple -42 doit être une valeur propre de l'Hamiltonien

eigenvalues = np.linalg.eigvals(hc(his, jijs, c, 4))

print(eigenvalues)


