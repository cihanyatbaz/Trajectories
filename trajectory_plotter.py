from Util import plotTrajectory
import pandas as pd

dataset = pd.read_csv("stitched_trajectories1.csv")

ids_list = dataset.Trajectory.unique()

for id in ids_list:
    trajectory = dataset[dataset.Trajectory == id]
    plotTrajectory(trajectory)