import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from numpy import linalg
import time

def createMatrixBroken(size, k): # fun funciton but does not produce the correct matrix
    arr = []
    for t in range(0, size):
        arr.append([])
        for x in range(0, size):
            arr[t].append([])
            for y in range(0, size):    
                arr[t][x].append(0)

        arr[t][t][t] = 1 + 4*k
        if (t == 0):
            arr[t][t+1][t] = -k
            arr[t][t][t+1] = -k
            continue

        if (t == size -1):
            arr[t][t-1][t] = -k
            arr[t][t][t-1] = -k
            continue
        
        arr[t][t-1][t] = -k
        arr[t][t+1][t] = -k
        arr[t][t][t-1] = -k
        arr[t][t][t+1] = -k
    return np.array(arr)

#------------------------------------------------
#------------------------------------------------ code below
def createMatrix(size, k):
    helpp = []
    for n in range(0, size):
        helpp.append([])
        for nn in range(0, size):
            helpp[n].append(nn+ size*n)

    arr = []
    s = size*size
    for y in range(0, s):
        arr.append([])
        b = findNeighbours(y, helpp, size)

        for x in range(0, s):
            if (x in b):
                arr[y].append(-k)
                continue
            arr[y].append(0)
            # walking across a x-file
        arr[y][y] = 4*k +1

    return np.array(arr)


def findNeighbours(x, helpMatrix, size):
    arr = []
    xxx = x - x%size

    x_comp = x - xxx
    y_comp = int(xxx/size)

    if (x_comp != 0 and x_comp != len(helpMatrix)-1 and y_comp != 0 and y_comp != len(helpMatrix)-1):
        arr.append(helpMatrix[y_comp-1][x_comp])
        arr.append(helpMatrix[y_comp+1][x_comp])
        arr.append(helpMatrix[y_comp][x_comp-1])
        arr.append(helpMatrix[y_comp][x_comp+1])
    elif (x_comp == 0 and y_comp == 0):
        arr.append(helpMatrix[y_comp+1][x_comp])
        arr.append(helpMatrix[y_comp][x_comp+1])
    elif (x_comp == len(helpMatrix)-1 and y_comp == len(helpMatrix)-1):
        arr.append(helpMatrix[y_comp-1][x_comp])
        arr.append(helpMatrix[y_comp][x_comp-1])
    elif (y_comp == 0 and x_comp == len(helpMatrix)-1):
        arr.append(helpMatrix[y_comp+1][x_comp])
        arr.append(helpMatrix[y_comp][x_comp-1])
    elif (x_comp == 0 and y_comp == len(helpMatrix)-1):
        arr.append(helpMatrix[y_comp-1][x_comp])
        arr.append(helpMatrix[y_comp][x_comp+1])
    elif (y_comp == 0 and x_comp != 0 and x_comp != len(helpMatrix)-1):
        arr.append(helpMatrix[y_comp+1][x_comp])
        arr.append(helpMatrix[y_comp][x_comp-1])
        arr.append(helpMatrix[y_comp][x_comp+1])
    elif (y_comp == len(helpMatrix)-1 and x_comp != 0 and x_comp != len(helpMatrix)-1):
        arr.append(helpMatrix[y_comp-1][x_comp])
        arr.append(helpMatrix[y_comp][x_comp-1])
        arr.append(helpMatrix[y_comp][x_comp+1])
    elif (x_comp == len(helpMatrix)-1 and y_comp != 0 and y_comp != len(helpMatrix)-1):
        arr.append(helpMatrix[y_comp-1][x_comp])
        arr.append(helpMatrix[y_comp+1][x_comp])
        arr.append(helpMatrix[y_comp][x_comp-1])
    elif (x_comp == 0 and y_comp != 0 and y_comp != len(helpMatrix)-1):
        arr.append(helpMatrix[y_comp-1][x_comp])
        arr.append(helpMatrix[y_comp+1][x_comp])
        arr.append(helpMatrix[y_comp][x_comp+1])

    return arr

def solveHeatEquationImplicit(init):
    iterations = 350
    x_nodes = len(init)
    dx = 1/x_nodes
    dt = 1/10000

    k_const = dt/(dx*dx)  #  factor

    start_time = time.time()
    A = createMatrix(x_nodes, k_const) # defines the matrix
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Matrix created. Time used: ", elapsed_time)

    start_time = time.time()
    inverse_A = linalg.inv(A)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Matrix inverted. Time used: ", elapsed_time)

    solution = []
    solution.append(np.array(init))

    for k in range(0, iterations):   # solves the system of equations for each timestep and adds it to the solution
        b = solution[k].flatten()   # flatten into a single array

        next_it = inverse_A.dot(b)

        next_it = next_it.reshape(x_nodes, x_nodes)  # reshape back into nxn array
        solution.append(next_it)
        print(k/iterations*100, " % complete")
    solution = np.array(solution)

        #-------------------------------- copy paste for creating this again
    def plotheatmap(u_k, k):
        # Clear the current plot figure
        plt.clf()

        plt.title(f"Temperature at t = {k*dt:.3f} unit time")
        plt.xlabel("x")
        plt.ylabel("y")

        # This is to plot u_k (u at time-step k)
        plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=-1, vmax=5) # adjust min/max values for colormap
        plt.colorbar()

        return plt

    def animate(k):
        plotheatmap(solution[k], k)
    #----------------------------------
    # the gif is created here 
    anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=iterations, repeat=False)
    anim.save("heat_equation_solution_implicit4.gif")

matrix_size = 80

# Generate a creative combination of patterns
x, y = np.meshgrid(np.linspace(0, 1, matrix_size), np.linspace(0, 1, matrix_size))

gradient = x + y
sinusoidal_pattern1 = 0.5 * np.sin(4 * np.pi * x)
sinusoidal_pattern2 = 0.3 * np.sin(8 * np.pi * y)
checkerboard_pattern = np.mod(np.floor(8 * x) + np.floor(8 * y), 2)
random_noise = np.random.uniform(-0.1, 0.1, size=(matrix_size, matrix_size))

initial_condition_matrix = gradient + sinusoidal_pattern1 + sinusoidal_pattern2 + checkerboard_pattern + random_noise


solveHeatEquationImplicit(initial_condition_matrix)
