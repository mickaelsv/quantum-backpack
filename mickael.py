class Knapsack:
    def __init__(self, max_volume, volume, utilities):
        self.max_volume = max_volume
        self.volume = volume
        self.utilities = utilities

def solve_knapsack(knapsack):
    num_items = len(knapsack.volume)
    dp_table = [[0 for _ in range(knapsack.max_volume + 1)] for _ in range(num_items + 1)]
    for item_index in range(1, num_items + 1):
        for current_volume in range(1, knapsack.max_volume + 1):
            if knapsack.volume[item_index - 1] <= current_volume:
                dp_table[item_index][current_volume] = max(
                    dp_table[item_index - 1][current_volume],
                    dp_table[item_index - 1][current_volume - knapsack.volume[item_index - 1]] + knapsack.utilities[item_index - 1]
                )
            else:
                dp_table[item_index][current_volume] = dp_table[item_index - 1][current_volume]

    # Check if the knapsack can be exactly filled
    if dp_table[num_items][knapsack.max_volume] == 0:
        return 0, []

    # Backtrack to find the items included in the knapsack
    selected_items = []
    remaining_volume = knapsack.max_volume
    for item_index in range(num_items, 0, -1):
        if dp_table[item_index][remaining_volume] != dp_table[item_index - 1][remaining_volume]:
            selected_items.append(item_index - 1)
            remaining_volume -= knapsack.volume[item_index - 1]

    # Ensure the knapsack is exactly full
    if remaining_volume != 0:
        return 0, []

    selected_items.reverse()
    # return the total volume, total volume, and item indexes
    return dp_table[num_items][knapsack.max_volume], selected_items, sum(knapsack.volume[i] for i in selected_items)

if __name__ == '__main__':
    knapsack = Knapsack(10, [2, 3, 4, 5], [3, 4, 5, 6])
    utility, items, volume = solve_knapsack(knapsack) 
    print("utility:", utility)
    print("items included:", items)
    print("total volume:", volume)


