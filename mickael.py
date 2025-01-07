class Knapsack:
    def __init__(self, max_weight, volume, utilities):
        self.max_weight = max_weight
        self.volume = volume
        self.utilities = utilities

def solve_knapsack(knapsack):
    n = len(knapsack.volume)
    dp = [[0 for _ in range(knapsack.max_weight + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, knapsack.max_weight + 1):
            if knapsack.volume[i - 1] <= j:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - knapsack.volume[i - 1]] + knapsack.utilities[i - 1])
            else:
                dp[i][j] = dp[i - 1][j]

    # Backtrack to find the items included in the knapsack
    selected_items = []
    w = knapsack.max_weight
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(i - 1)
            w -= knapsack.volume[i - 1]

    selected_items.reverse()
    return dp[n][knapsack.max_weight], selected_items


if __name__ == '__main__':
    knapsack = Knapsack(10, [2, 3, 4, 5], [3, 4, 5, 6])
    print(knapsack.max_weight)
    print(knapsack.volume)
    print(knapsack.utilities)
    print(solve_knapsack(knapsack))


