# Metaheuristic-Local_Search-Tabu_Search
Tabu Search Function for TSP problems. The function returns: 1) A list with the order of the cities to visit, and the total distance for visiting this same list order.

* X = Distance Matrix.

* buid_distance_matrix (HELPER FUNCTION) = Tranforms coordinates in a distance matrix (euclidean distance).

* city_tour = Initial list of visitation.

* seed (HELPER FUNCTION) = Generates a random list of visitation.

* iterations = Total number of iterations. The Default Value is 150.

* tabu_tenure = Maximum size of the Tabu List. The Default Value is 20.

* plot_tour_distance_matrix (HELPER FUNCTION) = A projection is generated based on the distance matrix. The estimated projection may present a plot with path crosses, even for the 2-opt optimal solution (Red Point = Initial city; Orange Point = Second City).

* plot_tour_coordinates (HELPER FUNCTION) = Plots the 2-opt optimal solution (Red Point = Initial city; Orange Point = Second City).

# Single Objective Optimization
For Single Objective Optimization try [pyMetaheuristic](https://github.com/Valdecy/pyMetaheuristic)

# Multiobjective Optimization or Many Objectives Optimization
For Multiobjective Optimization or Many Objectives Optimization try [pyMultiobjective](https://github.com/Valdecy/pyMultiobjective)

# TSP (Travelling Salesman Problem)
For Travelling Salesman Problems try [pyCombinatorial](https://github.com/Valdecy/pyCombinatorial)
