
class Knapsack:
    def __init__(self, volumes, utilities, capacity):
        self.volumes = volumes
        self.utilities = utilities
        self.capacity = capacity
        self.num_items = len(volumes)

def solve_knapsack(knapsack):
    n = knapsack.num_items
    V = knapsack.capacity

    # Initialiser la table DP : -1 signifie une solution invalide
    dp = [[-1] * (V + 1) for _ in range(n + 1)]
    dp[0][0] = 0  # On peut avoir un sac de poids 0 avec 0 utilité

    # Remplir la table DP
    for i in range(1, n + 1):
        for w in range(V + 1):
            # Si on ne prend pas l'objet i
            if dp[i - 1][w] != -1:
                dp[i][w] = max(dp[i][w], dp[i - 1][w])

            # Si on prend l'objet i
            if w >= knapsack.volumes[i - 1] and dp[i - 1][w - knapsack.volumes[i - 1]] != -1:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - knapsack.volumes[i - 1]] + knapsack.utilities[i - 1])

    # Si le sac ne peut pas être rempli exactement
    if dp[n][V] == -1:
        return None, None

    # Reconstruction de la solution
    solution = [0] * n
    w = V
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            solution[i - 1] = 1
            w -= knapsack.volumes[i - 1]

    return dp[n][V], solution


pb = Knapsack([4, 5, 5], [1000, 1, 1], 10)
print(solve_knapsack(pb))
