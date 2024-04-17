import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from numpy import linalg
import time
# Implisitt løsning av varmelikningen i to dimensjoner. Viste seg at jeg fikk til mer oppløsning på resultatet med denne løsningen. 
# Det kan ha noe med at jeg ikke fikk til å implementere parallell-kjøring av den eksplisitte metoden. Den store utfordingen med denne 
# framgangsmåten var alogritmen for å lage matrisen. Har lagt ved div kommentarer på spennende steder.

def createMatrixBroken(size, k): # første forsøk på å lage matrise. Funka ikke.
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
#------------------------------------------------ kode begynner her.
def createMatrix(size, k):
    helpp = []  # lage hjelpematrise av størrelse size * size. 
    for n in range(0, size):
        helpp.append([])
        for nn in range(0, size):
            helpp[n].append(nn+ size*n)

    arr = []    # lage faktisk matrise av størrelse size^2 * size^2. size^2 er antall piksler i beregningen vår så denne blir umennesklig
                # stor veldig fort. 
    s = size*size
    for y in range(0, s):
        arr.append([])
        b = findNeighbours(y, helpp, size) # hjelpefunksjon som returnerer array med posisjonene i arr som skal settes til -k. 

        for x in range(0, s):
            if (x in b): # sjekker om skal settes til -k eller 0.
                arr[y].append(-k)
                continue
            arr[y].append(0)
            # går langs x-linja
        arr[y][y] = 4*k +1

    return np.array(arr)


def findNeighbours(x, helpMatrix, size): # Finner naboer i hjelpematrisen og forteller oss hvor de vil legge seg i den store matrisen
                        # Hele hjelpematrisen blir til en enkelt rad i den store, og hver pixel representeres av sine naboer i en rad.
    arr = []
    xxx = x - x%size

    x_comp = x - xxx
    y_comp = int(xxx/size) # tror dette var noe omgjøring mellom koordinater. 

    if (x_comp != 0 and x_comp != len(helpMatrix)-1 and y_comp != 0 and y_comp != len(helpMatrix)-1): # lite elegant, men funker
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

def solveHeatEquationImplicit(init): # tar inn initialmatrise
    iterations = 350
    x_nodes = len(init)
    dx = 1/x_nodes
    dt = 1/10000

    k_const = dt/(dx*dx)  #  factor

    start_time = time.time()
    A = createMatrix(x_nodes, k_const) # lager matrisen A
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Matrix created. Time used: ", elapsed_time)

    start_time = time.time()
    inverse_A = linalg.inv(A) # Inverterer matrisen. Her tar ting lang tid, men sparer det igjen senere
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Matrix inverted. Time used: ", elapsed_time)

    solution = []
    solution.append(np.array(init))

    for k in range(0, iterations):   # Her lages løsninger
        b = solution[k].flatten()   # må gjøres fordi .dot() ikke er definert for n*n.

        next_it = inverse_A.dot(b)    # i stedet for linalg.solve(A, b)

        next_it = next_it.reshape(x_nodes, x_nodes)  # får tilbake orginal form
        solution.append(next_it)
        print(k/iterations*100, " % complete")
    solution = np.array(solution)

        #-------------------------------- copy paste
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
    # gif
    anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=iterations, repeat=False)
    anim.save("heat_equation_solution_implicit4.gif")

matrix_size = 80 # ikke sett noe særlig høyere.

# Genererer startbetingelse. (chatGPT som fant på)
x, y = np.meshgrid(np.linspace(0, 1, matrix_size), np.linspace(0, 1, matrix_size))

gradient = x + y
sinusoidal_pattern1 = 0.5 * np.sin(4 * np.pi * x)
sinusoidal_pattern2 = 0.3 * np.sin(8 * np.pi * y)
checkerboard_pattern = np.mod(np.floor(8 * x) + np.floor(8 * y), 2)
random_noise = np.random.uniform(-0.1, 0.1, size=(matrix_size, matrix_size))

initial_condition_matrix = gradient + sinusoidal_pattern1 + sinusoidal_pattern2 + checkerboard_pattern + random_noise


solveHeatEquationImplicit(initial_condition_matrix)
