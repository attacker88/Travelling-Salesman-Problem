# importing xml.etree to parse the xml file
import xml.etree.ElementTree as ET

# importing random to take random values
import random

# importing matplotlib to plot the convergence curve
import matplotlib.pyplot as plt


# Function to parse the xml file and then create a distance matrix
def create_distance_matrix(xml_file):
    # Parsing the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get the cities data from the xml file
    cities = []
    for city_elem in root.findall('.//vertex'):
        city = []
        for edge_elem in city_elem.findall('edge'):
            cost = float(edge_elem.get('cost'))
            city.append(cost)
        cities.append(city)
    print(cities)

    # Building the distance matrix
    num_cities = len(cities)
    distance_matrix = [[0.0 for _ in range(num_cities)] for _ in range(num_cities)]

    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            if i == j:
                distance_matrix[i][j] = 0.0
            else:
                distance_matrix[i][j] = cities[i][j - 1]
                distance_matrix[j][i] = cities[i][j - 1]
    print("Distance Matrix D: ", distance_matrix)
    return distance_matrix


# Brazil xml file path
brazil_xml_file = 'brazil58.xml'

# Call the distance matrix function and append the returned distance matrix to the variable
distance_matrix = create_distance_matrix(brazil_xml_file)

# Evolutionary Algorithm Parameters
population_size = 50
mutation_rate = 0.4
crossover_rate = 0.9
max_fitness_evaluations = 10000
tournament_size = 10

# fitness histories for each experiment
fitness_history1 = []
fitness_history2 = []
fitness_history3 = []
fitness_history4 = []

# number of cities
num_cities = len(distance_matrix)

# Generate random population
population = []


def initialize_population():
    for _ in range(population_size):
        individual = random.sample(range(num_cities), num_cities)
        population.append(individual)
    print("Random Population: ", population)


# Tournment selection to select parents
def tournament_selection(population_exp, tournament_size):
    tournament = random.sample(population_exp, tournament_size)
    best_individual = min(tournament, key=lambda x: calculate_total_distance(x))
    return best_individual


# Calculate the total cost for the tsp solution found
def calculate_total_distance(solution):
    total_distance = 0
    for i in range(num_cities):
        from_city = solution[i]
        to_city = solution[(i + 1) % num_cities]
        total_distance += distance_matrix[from_city][to_city]
    return total_distance


# Function to plot the convergence curve over the generations
def plot_convergence_curve(fitness_history, experiment_name):
    plt.figure()
    plt.plot(range(1, len(fitness_history) + 1), fitness_history)
    plt.xlabel("Evaluations")
    plt.ylabel("Best Fitness (Total Cost)")
    plt.title(f"Brazil - Convergence Curve for {experiment_name}")
    plt.grid(True)
    plt.show()


# Plot convergence curve for all experiments together
def convergence_curve_all_experiments(fitness_histories, experiment_names):
    plt.figure()
    for i in range(len(fitness_histories)):
        plt.plot(range(1, len(fitness_histories[i]) + 1), fitness_histories[i], label=experiment_names[i])

    plt.xlabel("Evaluations")
    plt.ylabel("Best Fitness (Total Cost)")
    plt.title("Brazil - Convergence Curves for all Experiments")
    plt.legend()
    plt.grid(True)
    plt.show()



# Experiment 1 with single point crossover and swap muatation
def experiment1():

    # Parameters required for the below operations
    population_exp1 = population.copy()
    best_cost = float('inf')
    fitness_evaluations = 0

    # Single point crossover function
    def single_point_crossover(a, b):
        point = random.randint(0, len(a) - 1)
        c = a[:point] + [gene for gene in b if gene not in a[:point]]
        d = b[:point] + [gene for gene in a if gene not in b[:point]]
        return c, d

    # Swap Mutation function
    def swap_mutation(individual):
        if random.random() < mutation_rate:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    # loop to evaluate fittness of the children and replace them if they are better
    while fitness_evaluations < max_fitness_evaluations:
        # Tournament Selection to select two parents
        a = tournament_selection(population_exp1, tournament_size)
        b = tournament_selection(population_exp1, tournament_size)

        # Single point crossover to generate two children
        c, d = single_point_crossover(a, b)

        # Mutation on children to generate new children
        e = swap_mutation(c)
        f = swap_mutation(d)

        # Calculate fitness of children
        fitness_child1 = calculate_total_distance(e)
        fitness_child2 = calculate_total_distance(f)

        # Replacement - Replacing the worst individuals
        worst_indices = sorted(range(population_size), key=lambda i: calculate_total_distance(population_exp1[i]))
        if fitness_child1 < calculate_total_distance(population_exp1[worst_indices[-2]]):
            population_exp1[worst_indices[-2]] = e
        if fitness_child2 < calculate_total_distance(population_exp1[worst_indices[-1]]):
            population_exp1[worst_indices[-1]] = f

        # storing best cost to plot the convergence curve
        if fitness_child1 < best_cost:
            best_cost = fitness_child1
        if fitness_child2 < best_cost:
            best_cost = fitness_child2

        fitness_history1.append(best_cost)

        fitness_evaluations += 2

    # Best solution and best cost
    best_solution = min(population_exp1, key=lambda x: calculate_total_distance(x))
    best_cost = calculate_total_distance(best_solution)

    # Printing and plotting the findings of the experiment
    print("Best solution experiment 1:", best_solution)
    print("Total Cost experiment 1:", best_cost)
    plot_convergence_curve(fitness_history1, "Experiment 1")

# Experiment 2 with single point crossover and inversion mutation
def experiment2():

    # Parameters required for the below operations
    population_exp2 = population.copy()
    best_cost = float('inf')
    fitness_evaluations = 0

    # Single point crossover function
    def single_point_crossover(a, b):
        point = random.randint(0, len(a) - 1)
        c = a[:point] + [gene for gene in b if gene not in a[:point]]
        d = b[:point] + [gene for gene in a if gene not in b[:point]]
        return c, d

    # Inversion mutation function
    def inversion_mutation(individual):
        #Selecting two distinct indices randomly
        i, j = random.sample(range(len(individual)), 2)
        if i > j:
            i, j = j, i

        # Reverse the order of genes between i and j
        individual[i:j + 1] = reversed(individual[i:j + 1])

        return individual

    # Loop to evaluate fittness of the children and replace them if they are better
    while fitness_evaluations < max_fitness_evaluations:
        # Tournament Selection to select two parents
        a = tournament_selection(population_exp2, tournament_size)
        b = tournament_selection(population_exp2, tournament_size)

        # Single point crossover to generate two children
        c, d = single_point_crossover(a, b)

        # Mutation on children
        e = inversion_mutation(c)
        f = inversion_mutation(d)

        # Calculate fitness of children
        fitness_child1 = calculate_total_distance(e)
        fitness_child2 = calculate_total_distance(f)

        # Replacement - Replacing the worst individuals
        worst_indices = sorted(range(population_size), key=lambda i: calculate_total_distance(population_exp2[i]))
        if fitness_child1 < calculate_total_distance(population_exp2[worst_indices[-2]]):
            population_exp2[worst_indices[-2]] = e
        if fitness_child2 < calculate_total_distance(population_exp2[worst_indices[-1]]):
            population_exp2[worst_indices[-1]] = f

        # storing best cost to plot the convergence curve
        if fitness_child1 < best_cost:
            best_cost = fitness_child1
        if fitness_child2 < best_cost:
            best_cost = fitness_child2

        fitness_history2.append(best_cost)

        fitness_evaluations += 2

    # Best solution and best cost
    best_solution = min(population_exp2, key=lambda x: calculate_total_distance(x))
    best_cost = calculate_total_distance(best_solution)

    # Printing and plotting the findings of the experiment
    print("Best solution experiment 2:", best_solution)
    print("Total Cost experiment 2:", best_cost)
    plot_convergence_curve(fitness_history2, "Experiment 2")

# Experiment 3 with ordered crossover and inversion mutation
def experiment3():

    # Parameters required for the below operations
    population_exp3 = population.copy()
    best_cost = float('inf')
    fitness_evaluations = 0

    # Ordered crossover function
    def ordered_crossover(a, b):
        start, end = sorted(random.sample(range(num_cities), 2))
        c = [-1] * num_cities
        d = [-1] * num_cities

        for i in range(start, end):
            c[i] = a[i]
            d[i] = b[i]

        pointer1, pointer2 = 0, 0
        for i in range(num_cities):
            if c[i] == -1:
                while b[pointer2] in c:
                    pointer2 += 1
                c[i] = b[pointer2]

            if d[i] == -1:
                while a[pointer1] in d:
                    pointer1 += 1
                d[i] = a[pointer1]

        return c, d

    #Inversion mutation function
    def inversion_mutation(individual):
        # Select two distinct indices randomly
        i, j = random.sample(range(len(individual)), 2)
        if i > j:
            i, j = j, i

        # Reverse the order of genes between i and j
        individual[i:j + 1] = reversed(individual[i:j + 1])

        return individual

    # Loop to evaluate fittness of the children and replace them if they are better
    while fitness_evaluations < max_fitness_evaluations:
        # Tournament Selection to select two parents
        a = tournament_selection(population_exp3, tournament_size)
        b = tournament_selection(population_exp3, tournament_size)

        # Ordered crossover to generate two children
        c, d = ordered_crossover(a, b)

        # Mutation on children
        e = inversion_mutation(c)
        f = inversion_mutation(d)

        # Calculate fitness of children
        fitness_child1 = calculate_total_distance(e)
        fitness_child2 = calculate_total_distance(f)

        # Replacement - Replacing the worst individuals
        worst_indices = sorted(range(population_size), key=lambda i: calculate_total_distance(population_exp3[i]))
        if fitness_child1 < calculate_total_distance(population_exp3[worst_indices[-2]]):
            population_exp3[worst_indices[-2]] = e
        if fitness_child2 < calculate_total_distance(population_exp3[worst_indices[-1]]):
            population_exp3[worst_indices[-1]] = f

        # storing best cost to plot the convergence curve
        if fitness_child1 < best_cost:
            best_cost = fitness_child1
        if fitness_child2 < best_cost:
            best_cost = fitness_child2

        fitness_history3.append(best_cost)

        fitness_evaluations += 2

    # Best solution and best cost
    best_solution = min(population_exp3, key=lambda x: calculate_total_distance(x))
    best_cost = calculate_total_distance(best_solution)

    # Printing and plotting the findings of the experiment
    print("Best solution experiment 3:", best_solution)
    print("Total Cost experiment 3:", best_cost)
    plot_convergence_curve(fitness_history3, "Experiment 3")

# Experiment 4 with ordered crossover and multiple swap mutation
def experiment4():

    # Parameters required for the below operations
    population_exp4 = population.copy()
    best_cost = float('inf')
    fitness_evaluations = 0

    # Ordered crossover function
    def ordered_crossover(a, b):
        start, end = sorted(random.sample(range(num_cities), 2))
        c = [-1] * num_cities
        d = [-1] * num_cities

        for i in range(start, end):
            c[i] = a[i]
            d[i] = b[i]

        pointer1, pointer2 = 0, 0
        for i in range(num_cities):
            if c[i] == -1:
                while b[pointer2] in c:
                    pointer2 += 1
                c[i] = b[pointer2]

            if d[i] == -1:
                while a[pointer1] in d:
                    pointer1 += 1
                d[i] = a[pointer1]

        return c, d

    # Multiple swap mutation function
    def multiple_swap_mutation(individual, num_swaps=1):
        num_cities = len(individual)

        if num_swaps < 1 or num_swaps > num_cities:
            raise ValueError("Invalid number of swaps")

        for _ in range(num_swaps):
            i, j = random.sample(range(num_cities), 2)
            individual[i], individual[j] = individual[j], individual[i]

        return individual

    # Loop to evaluate fittness of the children and replace them if they are better
    while fitness_evaluations < max_fitness_evaluations:
        # Tournament Selection to select two parents
        a = tournament_selection(population_exp4, tournament_size)
        b = tournament_selection(population_exp4, tournament_size)

        # Ordered crossover to generate two children
        c, d = ordered_crossover(a, b)

        # Mutation on children
        e = multiple_swap_mutation(c)
        f = multiple_swap_mutation(d)

        # Calculate fitness of children
        fitness_child1 = calculate_total_distance(e)
        fitness_child2 = calculate_total_distance(f)

        # Replacement - Replacing the worst individuals
        worst_indices = sorted(range(population_size), key=lambda i: calculate_total_distance(population_exp4[i]))
        if fitness_child1 < calculate_total_distance(population_exp4[worst_indices[-2]]):
            population_exp4[worst_indices[-2]] = e
        if fitness_child2 < calculate_total_distance(population_exp4[worst_indices[-1]]):
            population_exp4[worst_indices[-1]] = f

        # storing best cost to plot the convergence curve
        if fitness_child1 < best_cost:
            best_cost = fitness_child1
        if fitness_child2 < best_cost:
            best_cost = fitness_child2

        fitness_history4.append(best_cost)

        fitness_evaluations += 2

    # Best solution and best cost
    best_solution = min(population_exp4, key=lambda x: calculate_total_distance(x))
    best_cost = calculate_total_distance(best_solution)

    # Printing and plotting the findings of the experiment
    print("Best solution experiment 4:", best_solution)
    print("Total Cost experiment 4:", best_cost)
    plot_convergence_curve(fitness_history4, "Experiment 4")


# Calling the population and experiment functions
initialize_population()
experiment1()
experiment2()
experiment3()
experiment4()

convergence_curve_all_experiments([fitness_history1, fitness_history2, fitness_history3, fitness_history4],
                       ["Experiment 1", "Experiment 2", "Experiment 3", "Experiment 4"])