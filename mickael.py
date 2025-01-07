class Knapsack:
    def __init__(self, max_weight, volume, utilities):
        self.max_weight = max_weight
        self.volume = volume
        self.utilities = utilities

def solve_knapsack(knapsack):
    num_items = len(knapsack.volume)
    dp_table = [[0 for _ in range(knapsack.max_weight + 1)] for _ in range(num_items + 1)]
    for item_index in range(1, num_items + 1):
        for current_weight in range(1, knapsack.max_weight + 1):
            if knapsack.volume[item_index - 1] <= current_weight:
                dp_table[item_index][current_weight] = max(
                    dp_table[item_index - 1][current_weight],
                    dp_table[item_index - 1][current_weight - knapsack.volume[item_index - 1]] + knapsack.utilities[item_index - 1]
                )
            else:
                dp_table[item_index][current_weight] = dp_table[item_index - 1][current_weight]

    # Backtrack to find the items included in the knapsack
    selected_items = []
    remaining_weight = knapsack.max_weight
    for item_index in range(num_items, 0, -1):
        if dp_table[item_index][remaining_weight] != dp_table[item_index - 1][remaining_weight]:
            selected_items.append(item_index - 1)
            remaining_weight -= knapsack.volume[item_index - 1]

    selected_items.reverse()
    return dp_table[num_items][knapsack.max_weight], selected_items

if __name__ == '__main__':
    knapsack = Knapsack(10, [2, 3, 4, 5], [3, 4, 5, 6])
    print(knapsack.max_weight)
    print(knapsack.volume)
    print(knapsack.utilities)
    print(solve_knapsack(knapsack))


