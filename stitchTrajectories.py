import pandas as pd
import numpy as np
import Util




def checkAndStitchTrajectories(t1, t2, threshold = 0.5):
    # procedure that checks the compatibility between two trajectories and if they are compatible merges them

    t1_last_positions = t1.tail(2)[["X (pixel)","Y (pixel)"]].values
    t2_first_positions = t2.head(2)[["X (pixel)","Y (pixel)"]].values

    # pos1_t1 and pos2_t1 represent the last 2 positions in trajectory t1
    # pos1_t2 and pos2_t2 represent the first 2 positions in trajectory t2
    pos1_t1 = t1_last_positions[0]
    pos2_t1 = t1_last_positions[1]
    pos1_t2 = t2_first_positions[0]
    pos2_t2 = t2_first_positions[1]

    # difference vector between pos2_t1 and pos1_t1, represents the movement between the 2 positions
    vect1 = np.subtract(pos2_t1, pos1_t1)
    # difference vector between pos1_t2 and pos2_t1
    vect2 = np.subtract(pos1_t2, pos2_t1)
    # difference vector between pos2_t2 and pos1_t2
    vect3 = np.subtract(pos2_t2, pos1_t2)

    # ratio between the euclidean distance of vect1 and vect2 and the shortest of the 2 vectors
    # basically it indicates how much the movement between the last 2 positions of trajectory 1 and the movement
    # from last position of t1 to the first position of t2 differ
    #intensity_change_1 = (Util.euclideanDistance(vect1, vect2) / min(Util.vectLen(vect1),Util.vectLen(vect2)))
    # ratio between the euclidean distance of vect2 and vect3 and the shortest of the 2 vectors
    # indicates how much the movement from the last of trajectory 1 and the first position of t2 and the movement
    # between the first 2 positions of t2 differ
    #intensity_change_2 = (Util.euclideanDistance(vect2, vect3)/ min(Util.vectLen(vect3),Util.vectLen(vect2)))

    intensity_change_11 = max(Util.vectLen(vect1),Util.vectLen(vect2)) / min(Util.vectLen(vect1),Util.vectLen(vect2))
    intensity_change_21 = max(Util.vectLen(vect3),Util.vectLen(vect2)) / min(Util.vectLen(vect3),Util.vectLen(vect2))

    # cosine distance between vect1 and vect2
    similarity1 = Util.getSimilarity(vect1,vect2)
    # cosine distance between vect2 and vect3
    similarity2 = Util.getSimilarity(vect3, vect2)

    # score combining the movement differences and the cosine distance
    score1 = similarity1 / intensity_change_11
    score2 = similarity2 / intensity_change_21

    # basically if both vect1 and vect3 (respectively the movement between the last 2 positions of t1 and
    # the movement between the first 2 positions of t2) are similar, to a certain degree, to vect2 (representing
    # the movement from the last position of t1 to the first position of t2), the 2 trajectories are considered
    # compatible
    if score1 > threshold and score2 > threshold:
        print("Scores: ")
        print(t1["Trajectory"].values[0])
        print(t2["Trajectory"].values[0])
        print(score1)
        print(score2)

        new_t = pd.concat([t1,t2])
        return new_t
        #plotTrajectory(new_t)

    return None
        # print(score1)
        # print(score2)
        # print(t1.tail(2))
        # print(t2.head(2))

def iterateTrajectories(t_df1, t_df2= None, iteration = 1):
    # function that tries to stitch trajectories from t_df1 to trajectories of t_df2

    if t_df2 is None: t_df2 = t_df1.copy()
    # take list of all trajectory ids in both dataframes
    t_ids_list1 = t_df1.Trajectory.unique()
    t_ids_list2 = t_df2.Trajectory.unique()
    print("iteration: " + str(iteration))

    # inizialize list of stitched trajectories
    stitched_trajectories_list = []
    # iterate trough trajectories in the first dataframe
    for t_id1 in t_ids_list1:
        # get first trajectory
        trajectory1 = t_df1[t_df1.Trajectory == t_id1]
        # get last frame of first trajectory
        last_frame = trajectory1.tail(1)["Frame"].values[0]

        # iterate trough trajectories in the second dataframe
        for t_id2 in t_ids_list2:
            # get second trajectory
            trajectory2 = t_df2[t_df2.Trajectory == t_id2]

            # checks if the first frame of the second trajectory equals the last frame of the first trajectory + 1
            if trajectory2.head(1)["Frame"].values[0] == last_frame + 1:
                # call the stitching procedure
                new_trajectory = checkAndStitchTrajectories(trajectory1, trajectory2)

                # if the 2 trajectories have been stitched
                if new_trajectory is not None:
                    # add the new trajectory to the list
                    new_trajectory["Trajectory"] = max(t_ids_list1) + 1 + len(stitched_trajectories_list)
                    print(new_trajectory)
                    stitched_trajectories_list.append(new_trajectory)

    # if at least one new trajectory has been found call the function recursively by using the new trajectories
    # as t_df1 and by keeping t_df2 the same
    # also save the new trajectories on a file
    if stitched_trajectories_list:
        stiched_trajectories_df = pd.concat(stitched_trajectories_list)
        stiched_trajectories_df.to_csv("stiched_trajectories.csv"+str(iteration), index=False)
        iterateTrajectories(stiched_trajectories_df, t_df2, iteration+1)
    else:
        print("stiching completed")
        return



if __name__ == '__main_':
    dataset = pd.read_csv("trajectories.csv")
    header = dataset.columns[0:4].values
    X = dataset.iloc[:, 0:4].values  # select columns 1 through 3 included

    X_DF = pd.DataFrame(X, columns=header)
    n_trajectories = X_DF.nunique()[0]

    iterateTrajectories(X_DF)
