import grover
import knapsack
import time
import QAOA
import numpy as np


def test_sack(sack):
    start=time.time()
    result_state, result_utility = grover.solve_knapsack_optimization(sack)
    end=time.time()
    print("Grover solution : \n")
    print("\ttime :",end-start,"s")
    print(f"\tSelected State: {result_state}, Total Utility: {result_utility}")

    start=time.time()
    # def QAOA(volumes, utilities, volume, n, l, reps):
    solution,utilities = QAOA.QAOA(np.array(sack.volume),np.array(sack.utilities),sack.max_volume,len(sack.utilities),sum(sack.utilities)+1,10)
    end=time.time()
    print("\nQAOA solution : \n")
    print("\ttime :",end-start,"s")
    print("\tSelected State:", solution, "Total Utility:" ,utilities )


    start=time.time()
    max_value,solution  = knapsack.solve_knapsack(sack)
    end=time.time()
    print("\nBruteforce solution : \n")
    print("\ttime :",end-start,"s")
    print("\tSelected State:", solution, "Total Utility:" ,max_value )

def main_test():
    sack1 =  knapsack.Knapsack(2,[2, 1, 1],[1, 2, 1])
    sack2 =  knapsack.Knapsack(5,[2, 1, 1,3,4],[1, 2, 1,2,5])
    sack3 =  knapsack.Knapsack(30,[2, 1, 1,3,4,7,10,2,3],[1, 2, 1,2,5,15,32,1,5])
    test_sack(sack1)
    print("####################")
    test_sack(sack2)
    print("####################")
    test_sack(sack3)

main_test()

