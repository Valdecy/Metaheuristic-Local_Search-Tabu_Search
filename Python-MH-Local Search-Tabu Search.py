############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: Local Search-Tabu Search

# Citation: 
# PEREIRA, V. (2018). Project: Metaheuristic-Local_Search-Tabu_Search, File: Python-MH-Local Search-Tabu Search.py, GitHub repository: <https://github.com/Valdecy/Metaheuristic-Local_Search-Tabu_Search>

############################################################################

# Required Libraries
import pandas as pd
import random
import numpy  as np
import copy
from matplotlib import pyplot as plt 

# Function: Tour Distance
def distance_calc(Xdata, city_tour):
    distance = 0
    for k in range(0, len(city_tour[0])-1):
        m = k + 1
        distance = distance + Xdata.iloc[city_tour[0][k]-1, city_tour[0][m]-1]            
    return distance

# Function: Euclidean Distance 
def euclidean_distance(x, y):       
    distance = 0
    for j in range(0, len(x)):
        distance = (x.iloc[j] - y.iloc[j])**2 + distance   
    return distance**(1/2) 

# Function: Initial Seed
def seed_function(Xdata):
    seed = [[],float("inf")]
    sequence = random.sample(list(range(1,Xdata.shape[0]+1)), Xdata.shape[0])
    sequence.append(sequence[0])
    seed[0] = sequence
    seed[1] = distance_calc(Xdata, seed)
    return seed

# Function: Build Distance Matrix
def buid_distance_matrix(coordinates):
    Xdata = pd.DataFrame(np.zeros((coordinates.shape[0], coordinates.shape[0])))
    for i in range(0, Xdata.shape[0]):
        for j in range(0, Xdata.shape[1]):
            if (i != j):
                x = coordinates.iloc[i,:]
                y = coordinates.iloc[j,:]
                Xdata.iloc[i,j] = euclidean_distance(x, y)        
    return Xdata

# Function: Tour Plot
def plot_tour_distance_matrix (Xdata, city_tour):
    m = Xdata.copy(deep = True)
    for i in range(0, Xdata.shape[0]):
        for j in range(0, Xdata.shape[1]):
            m.iloc[i,j] = (1/2)*(Xdata.iloc[0,j]**2 + Xdata.iloc[i,0]**2 - Xdata.iloc[i,j]**2)    
    m = m.values
    w, u = np.linalg.eig(np.matmul(m.T, m))
    s = (np.diag(np.sort(w)[::-1]))**(1/2) 
    coordinates = np.matmul(u, s**(1/2))
    coordinates = coordinates.real[:,0:2]
    xy = pd.DataFrame(np.zeros((len(city_tour[0]), 2)))
    for i in range(0, len(city_tour[0])):
        if (i < len(city_tour[0])):
            xy.iloc[i, 0] = coordinates[city_tour[0][i]-1, 0]
            xy.iloc[i, 1] = coordinates[city_tour[0][i]-1, 1]
        else:
            xy.iloc[i, 0] = coordinates[city_tour[0][0]-1, 0]
            xy.iloc[i, 1] = coordinates[city_tour[0][0]-1, 1]
    plt.plot(xy.iloc[:,0], xy.iloc[:,1], marker = 's', alpha = 1, markersize = 7, color = 'black')
    plt.plot(xy.iloc[0,0], xy.iloc[0,1], marker = 's', alpha = 1, markersize = 7, color = 'red')
    plt.plot(xy.iloc[1,0], xy.iloc[1,1], marker = 's', alpha = 1, markersize = 7, color = 'orange')
    return

# Function: Tour Plot
def plot_tour_coordinates (coordinates, city_tour):
    coordinates = coordinates.values
    xy = pd.DataFrame(np.zeros((len(city_tour[0]), 2)))
    for i in range(0, len(city_tour[0])):
        if (i < len(city_tour[0])):
            xy.iloc[i, 0] = coordinates[city_tour[0][i]-1, 0]
            xy.iloc[i, 1] = coordinates[city_tour[0][i]-1, 1]
        else:
            xy.iloc[i, 0] = coordinates[city_tour[0][0]-1, 0]
            xy.iloc[i, 1] = coordinates[city_tour[0][0]-1, 1]
    plt.plot(xy.iloc[:,0], xy.iloc[:,1], marker = 's', alpha = 1, markersize = 7, color = 'black')
    plt.plot(xy.iloc[0,0], xy.iloc[0,1], marker = 's', alpha = 1, markersize = 7, color = 'red')
    plt.plot(xy.iloc[1,0], xy.iloc[1,1], marker = 's', alpha = 1, markersize = 7, color = 'orange')
    return

# Function:  Build Recency Based Memory and Frequency Based Memory (STM and LTM)
def build_stm_and_ltm(Xdata):
    n = int((Xdata.shape[0]**2 - Xdata.shape[0])/2)
    stm_and_ltm = pd.DataFrame(np.zeros((n, 5)), columns = ['City 1','City 2','Recency', 'Frequency', 'Distance'])
    count = 0
    for i in range (0, int((Xdata.shape[0]**2))):
        city_1 = i // (Xdata.shape[1])
        city_2 = i %  (Xdata.shape[1])
        if (city_1 < city_2):
            stm_and_ltm.iloc[count, 0] = city_1 + 1
            stm_and_ltm.iloc[count, 1] = city_2 + 1
            count = count + 1
    return stm_and_ltm

# Function: Swap
def local_search_2_swap(Xdata, city_tour, m, n):
    best_route = copy.deepcopy(city_tour)       
    best_route[0][m], best_route[0][n] = best_route[0][n], best_route[0][m]        
    best_route[0][-1]  = best_route[0][0]              
    best_route[1] = distance_calc(Xdata, best_route)                     
    city_list = copy.deepcopy(best_route)         
    return city_list
	
# Function: 2_opt
def local_search_2_opt(Xdata, city_tour):
    city_list = copy.deepcopy(city_tour)
    best_route = copy.deepcopy(city_list)
    seed = copy.deepcopy(city_list)        
    for i in range(0, len(city_list[0]) - 2):
        for j in range(i+1, len(city_list[0]) - 1):
            best_route[0][i:j+1] = list(reversed(best_route[0][i:j+1]))           
            best_route[0][-1]  = best_route[0][0]                          
            best_route[1] = distance_calc(Xdata, best_route)    
            if (best_route[1] < city_list[1]):
                city_list[1] = copy.deepcopy(best_route[1])
                for n in range(0, len(city_list[0])): 
                    city_list[0][n] = best_route[0][n]          
            best_route = copy.deepcopy(seed) 
    return city_list

# Function: Diversification
def ltm_diversification (Xdata, stm_and_ltm, city_list):
    stm_and_ltm = stm_and_ltm.sort_values(['Frequency', 'Distance'], ascending = [True, True])
    lenght = random.sample((range(1, int(Xdata.shape[0]/3))), 1)[0]
    for i in range(0, lenght):
        m = int(stm_and_ltm.iloc[i, 0] - 1)
        n = int(stm_and_ltm.iloc[i, 1] - 1)
        city_list = local_search_2_swap(Xdata, city_list, m, n)
        stm_and_ltm.iloc[i, 3] = stm_and_ltm.iloc[i, 3] + 1
        stm_and_ltm.iloc[i, 2] = 1
    return stm_and_ltm, city_list

# Function: 4 opt Stochastic
def local_search_4_opt_stochastic(Xdata, city_tour):
    best_route = copy.deepcopy(city_tour)
    best_route_03 = [[],float("inf")]
    best_route_04 = [[],float("inf")]
    best_route_11 = [[],float("inf")]
    best_route_22 = [[],float("inf")]
    best_route_27 = [[],float("inf")] 
    i, j, k, L = np.sort(random.sample(list(range(0,Xdata.shape[0])), 4))                                
    best_route_03[0] = best_route[0][:i+1] + best_route[0][k+1:L+1] + best_route[0][j+1:k+1] + best_route[0][i+1:j+1] + best_route[0][L+1:]
    best_route_03[1] = distance_calc(Xdata, best_route_03) # ADCB                      
    best_route_04[0] = best_route[0][:i+1] + list(reversed(best_route[0][i+1:j+1])) + best_route[0][j+1:k+1] + list(reversed(best_route[0][k+1:L+1])) + best_route[0][L+1:]                  
    best_route_04[1] = distance_calc(Xdata, best_route_04)  # AbCd
    best_route_11[0] = best_route[0][:i+1] + best_route[0][k+1:L+1] + list(reversed(best_route[0][i+1:j+1])) + list(reversed(best_route[0][j+1:k+1])) + best_route[0][L+1:]
    best_route_11[1] = distance_calc(Xdata, best_route_11)   # ADbc                                          
    best_route_22[0] = best_route[0][:i+1] + list(reversed(best_route[0][j+1:k+1])) + list(reversed(best_route[0][k+1:L+1])) + best_route[0][i+1:j+1] + best_route[0][L+1:]
    best_route_22[1] = distance_calc(Xdata, best_route_22) # AcdB                       
    best_route_27[0] = best_route[0][:i+1] + list(reversed(best_route[0][k+1:L+1])) + best_route[0][j+1:k+1] + list(reversed(best_route[0][i+1:j+1])) + best_route[0][L+1:]
    best_route_27[1] = distance_calc(Xdata, best_route_27) # AdCb    
    best_route = copy.deepcopy(best_route_03)          
    if(best_route_04[1]  < best_route[1]):
        best_route = copy.deepcopy(best_route_04)
    elif(best_route_11[1]  < best_route[1]):
        best_route = copy.deepcopy(best_route_11)            
    elif(best_route_22[1]  < best_route[1]):
        best_route = copy.deepcopy(best_route_22)            
    elif(best_route_27[1]  < best_route[1]):
        best_route = copy.deepcopy(best_route_27)          
    return best_route	

# Function: Tabu Update
def tabu_update(Xdata, stm_and_ltm, city_list, best_distance, tabu_list, tabu_tenure = 20, diversify = False):
    m_list = []
    n_list = []
    city_list = local_search_2_opt(Xdata, city_list) # itensification
    for i in range(0, stm_and_ltm.shape[0]):
        m = int(stm_and_ltm.iloc[i, 0] - 1)
        n = int(stm_and_ltm.iloc[i, 1] - 1)
        stm_and_ltm.iloc[i, -1] = local_search_2_swap(Xdata, city_list, m, n)[1] 
    stm_and_ltm = stm_and_ltm.sort_values(by = 'Distance')
    m = int(stm_and_ltm.iloc[0,0]-1)
    n = int(stm_and_ltm.iloc[0,1]-1)
    recency = int(stm_and_ltm.iloc[0,2])
    distance = stm_and_ltm.iloc[0,-1]     
    if (distance < best_distance): # Aspiration Criterion -> by Objective
        city_list = local_search_2_swap(Xdata, city_list, m, n)
        i = 0
        while (i < stm_and_ltm.shape[0]):
            if (stm_and_ltm.iloc[i, 0] == m + 1 and stm_and_ltm.iloc[i, 1] == n + 1):
                stm_and_ltm.iloc[i, 2] = 1
                stm_and_ltm.iloc[i, 3] = stm_and_ltm.iloc[i, 3] + 1
                stm_and_ltm.iloc[i, -1] = distance
                if (stm_and_ltm.iloc[i, 3] == 1):
                    m_list.append(m + 1)
                    n_list.append(n + 1)
                i = stm_and_ltm.shape[0]
            i = i + 1
    else:
        i = 0
        while (i < stm_and_ltm.shape[0]):
            m = int(stm_and_ltm.iloc[i,0]-1)
            n = int(stm_and_ltm.iloc[i,1]-1)
            recency = int(stm_and_ltm.iloc[i,2]) 
            distance = local_search_2_swap(Xdata, city_list, m, n)[1]
            if (distance < best_distance):
                city_list = local_search_2_swap(Xdata, city_list, m, n)
            if (recency == 0):
                city_list = local_search_2_swap(Xdata, city_list, m, n)
                stm_and_ltm.iloc[i, 2] = 1
                stm_and_ltm.iloc[i, 3] = stm_and_ltm.iloc[i, 3] + 1
                stm_and_ltm.iloc[i, -1] = distance
                if (stm_and_ltm.iloc[i, 3] == 1):
                    m_list.append(m + 1)
                    n_list.append(n + 1)
                i = stm_and_ltm.shape[0]
            i = i + 1
    if (len(m_list) > 0): 
        tabu_list[0].append(m_list[0])
        tabu_list[1].append(n_list[0])
    if (len(tabu_list[0]) > tabu_tenure):
        i = 0
        while (i < stm_and_ltm.shape[0]):
            if (stm_and_ltm.iloc[i, 0] == tabu_list[0][0] and stm_and_ltm.iloc[i, 1] == tabu_list[1][0]):
                del tabu_list[0][0]
                del tabu_list[1][0]
                stm_and_ltm.iloc[i, 2] = 0
                i = stm_and_ltm.shape[0]
            i = i + 1          
    if (diversify == True):
        stm_and_ltm, city_list = ltm_diversification(Xdata, stm_and_ltm, city_list) # diversification
        city_list = local_search_4_opt_stochastic(Xdata, city_list) # diversification
    return stm_and_ltm, city_list, tabu_list

# Function: Tabu Search
def tabu_search(Xdata, city_tour, iterations = 150, tabu_tenure = 20):
    count = 0
    best_solution = copy.deepcopy(city_tour)
    stm_and_ltm = build_stm_and_ltm(Xdata)
    tabu_list = [[],[]]
    diversify = False
    no_improvement = 0
    while (count < iterations):       
        stm_and_ltm, city_tour, tabu_list = tabu_update(Xdata, stm_and_ltm, city_tour, best_solution[1], tabu_list = tabu_list, tabu_tenure = tabu_tenure, diversify = diversify)
        if (city_tour[1] < best_solution[1]):
            best_solution = copy.deepcopy(city_tour)
            no_improvement = 0
            diversify = False
        else:
            no_improvement = no_improvement + 1
            if (no_improvement > int(iterations/5)):
                diversify = True
        count = count + 1
        print("Iteration =", count, "-> Distance =", best_solution[1])
    print("Best Solution =", best_solution)
    return best_solution

######################## Part 1 - Usage ####################################

X = pd.read_csv('Python-MH-Local Search-Tabu Search-Dataset-01.txt', sep = '\t') # 17 cities = 1922.33
seed = seed_function(X)
lsts = tabu_search(X, city_tour = seed, iterations = 50, tabu_tenure = 7)
plot_tour_distance_matrix(X, lsts) # Red Point = Initial city; Orange Point = Second City # The generated coordinates (2D projection) are aproximated, depending on the data, the optimum tour may present crosses.

Y = pd.read_csv('Python-MH-Local Search-Tabu Search-Dataset-02.txt', sep = '\t') # Berlin 52 = 7544.37
X = buid_distance_matrix(Y)
seed = seed_function(X)
lsts = tabu_search(X, city_tour = seed, iterations = 100, tabu_tenure = 7)
plot_tour_coordinates (Y, lsts) # Red Point = Initial city; Orange Point = Second City
