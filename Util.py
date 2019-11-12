import math
import pandas as pd

def euclideanDistance(pos1, pos2):
    return math.sqrt(pow((pos1[0]-pos2[0]),2) + pow((pos1[1]-pos2[1]),2))

def dotProduct(a,b):
    result = 0
    if len(a) == len(b):
        for i in range(len(a)):
            result += a[i] * b[i]
        return result
    else:
        return -1

def vectLen(a):
    val = dotProduct(a,a)

    return math.sqrt(val)

def getSimilarity(a,b):
    return dotProduct(a,b) / (vectLen(a) * vectLen(b))

def plotTrajectory(T):
    # plotting trajectory T
    import matplotlib
    matplotlib.use("TKAgg")
    from matplotlib import pyplot as plt

    X = [int(elem) for elem in T["X (pixel)"].tolist()]
    Y = [int(elem) for elem in T["Y (pixel)"].tolist()]
    plt.scatter(X, Y, cmap="Paired")
    plt.title(str(T["Trajectory"].values[0]) + "and" + str(T["Trajectory"].values[-1]))
    plt.show()