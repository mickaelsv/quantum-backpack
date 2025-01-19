
class Knapsack:
    """
    Class to represent a knapsack problem.
    """
    def __init__(self, max_volume, volume, utilities):
        self.max_volume = max_volume
        self.volume = volume
        self.utilities = utilities


def solve_knapsack(knapsack):
    n = len(knapsack.volume)
    V = knapsack.max_volume

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
            if w >= knapsack.volume[i - 1] and dp[i - 1][w - knapsack.volume[i - 1]] != -1:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - knapsack.volume[i - 1]] + knapsack.utilities[i - 1])

    # Si le sac ne peut pas être rempli exactement
    if dp[n][V] == -1:
        return None, None

    # Reconstruction de la solution
    solution = [0] * n
    w = V
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            solution[i - 1] = 1
            w -= knapsack.volume[i - 1]

    return dp[n][V], solution

if __name__ == "__main__":
    pb = Knapsack(10, [4, 5, 5], [1000, 1, 1])
    print(solve_knapsack(pb))
