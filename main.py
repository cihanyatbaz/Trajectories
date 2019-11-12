import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import Util

def create_dataset(dataset, look_back=1):
    # convert an array of coordinates into 2 np arrays:
    # dataX containing the coordinates to be used as independent variables grouped in sets of "look_back" size
    # dataY containing the real target coordinates
    # dataset = dataframe containing data regarding one trajectory
    # lookback = how many 'past' coordinates we want to use to predict the next one
    # (e.g lookback = k then to predict coordinate at time t we use coordinates at times t-1, t-2, ..., t-k)

    # initializing lists that will contain the transformed data
    dataX, dataY = [], []

    # looping through the trajectory coordinates
    for i in range(len(dataset) - look_back):
        # grouping the coordinates in sets of size k = look_back
        # (e.g. [[c1,c2,..,ck], [c2,c3,..,ck+1],....,[cn-2,cn-1,c-n]],
        # n = total number of coordinates of the current trajectory)
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        # for each group of coordinates insert in the list the target coordinate
        # if Xdata[i] = [ci,ci+1,..,ck+i-1] -> Ydata =[cK+i]
        dataY.append(dataset[i + look_back, :])

    # returning the 2 lists
    return np.array(dataX), np.array(dataY)

def split_dataset(dataset, test_size, random_state = None):
    # given the trajetories dataset split it into training and test data
    # dataset = dataset to split
    # test_size = [0,1] fraction of the dataset to use as test_set
    # random_state = defines the seed for the random selection

    # retrieving the list of all unique ids in the dataset
    trajectory_ids = np.unique(dataset[:,0])

    # splitting the ids into 2 sets: training_ids and test_ids, identifying respectevly the ids of the
    # trajectories that will be placed in the training set and of those that will be placed in hte
    from sklearn.model_selection import train_test_split
    train_ids, test_ids = train_test_split(trajectory_ids, test_size=test_size, random_state=random_state)

    # defining training and test sets based on the respective ids
    training_set = np.asarray([elem for elem in dataset if elem[0] in train_ids])
    test_set = np.asarray([elem for elem in dataset if elem[0] in test_ids])

    # returning training and test sets
    return training_set, test_set

def define_model(look_back):
    # defining the lstm model
    # look_back = number of coordinates to be used as training, used to define the input_shape for the hidden layer
    from keras.models import Sequential
    from keras.layers import LSTM
    from keras.layers import Dense

    # defining a sequential model
    model = Sequential()
    # adding one LSTM layer with 4 nodes to the model (hidden), to which we will feed, as input, k = look_back coordinates
    model.add(LSTM(4, input_shape=(look_back, 2)))
    # adding one Dense layer with 2 nodes to the model (output), which will return the predicted coordinate
    model.add(Dense(2))
    #model.add(LSTM(2,return_sequences=True))
    #model.add(TimeDistributed(Dense(2, activation='sigmoid')))
    #model.compile(loss='mse', optimizer='adam')
    # determining which loss function and optimizer to use
    model.compile(loss="mse", optimizer='Adam')

    # printing a summary of the model
    model.summary()

    # returning the model
    return model

def train_model(model, dataset, look_back, save_model = False, filename = "test.h5"):
    # training the model over each trajectory in the training set, the training is done by using 'look_back' consecutive coordinates
    # as input and comparing the result with the 'train_size + 1'th consecutive coordinate (which represents the prediction)
    # model = model to be trained
    # dataset = training set for the model
    # look_back = number of coordinates used to make a prediction
    # save_model = if True the model weights get saved on a file
    # filename = file to which the weights will be saved

    # retrieving all the unique ids in the training set
    train_ids = np.unique(dataset[:,0])

    # looping through all the trajectories of the training set
    for count,id in enumerate(train_ids):
        print("fitting model to " + str(count) +"th trajectory")

        # taking the coordinate (x,y)  columns of all the rows concerning the trajectory with the current id
        current_t_coordinates = np.asarray([elem[[2,3]] for elem in dataset if elem[0] == id])

        # producing the dataset so that the prediction is done on 'train size' coordinates
        X,y = create_dataset(current_t_coordinates, look_back)

        # if the current trajectory has less then training_size coordinates don't consider it
        if X.shape[0] == 0: continue

        # reshaping the model input, correct shape for the model input = (batch size, timelength, features)
        X = np.reshape(X, (X.shape[0], look_back, 2))

        # training the model with the data of the current trajectory
        # epochs = number of times the model is trained on the same trajectory
        # batch size = number of training steps before the model gets updated
        model.fit(X,y,epochs=50, batch_size=X.shape[0], verbose=0)

        # used for testing
        #if count >=100: break

    if save_model:
        model.save_weights(filename)

def evaluate_model(model, test_set, look_back, prediction_steps = 1, plot_data = False):
    # computing the mse for the model on prediction_steps predictions
    # model = model to be evaluated
    # test_set = dataset used for the evalution
    # look_back = number of coordinates to be used for prediction
    # prediction_steps = number of steps the model has to predict
    # plot_data = if True, plors the trajectories predicted by the model

    # retrieving all the ids of the trajectories in the test set
    test_ids = np.unique(test_set[:, 0])

    # initializing lists that will contain respectively all predicted and real predictions
    # to be used to compute mse
    all_predicted = []
    all_real = []

    # looping through all the trajectories in the test set
    for id in test_ids:
        # extracting coordinates of the trajectory with the current id
        current_test_t = np.asarray([elem[[2, 3]] for elem in test_set if elem[0] == id])

        # if the total number of coordinates of the current trajectory is too low to be used for prediction
        # skip it
        if len(current_test_t) < look_back + prediction_steps: continue

        # formatting the data to be used for the predicitons
        X_test, y_test = create_dataset(current_test_t, look_back)

        # initializing the list of predicted trajectoried (used only for plotting)
        Yhat = []

        # looping through the coordinates of the current trajectory
        for i in range(X_test.shape[0]-prediction_steps+1):
            # retrieving the first group of k = look_back coordinates to use as the baseline for
            # the 'prediction_step' predictions to be performed
            baseline=X_test[i,:]

            # initializing the list of predictions
            lookahead_predictions = []

            # predicting the next n = 'prediction_step' coordinates after the baseline ones
            for step in range(prediction_steps):
                if lookahead_predictions:
                    # if this is the ith > 1 prediction use as input for the prediction the coordinates contained in the
                    # baseline list from index i to the end of the list combined with the coordinates predicted in the
                    # previous steps

                    # A = look_back - i elements, at the ith step
                    baseline_coords_left = baseline[step:,:]
                    # prediction_steps - A elements
                    previous_predictions = np.array(lookahead_predictions)[max(0,step-look_back):step+1,:]
                    my_input = np.concatenate((baseline_coords_left, previous_predictions))
                else:
                    # if this is the 1st prediction just use the baseline coordinates as input for the model
                    my_input = baseline[step:,:]

                # reshape the input to fit the appropriate shape for the model
                my_input=np.reshape(my_input,(1,my_input.shape[0],my_input.shape[1]))

                # predict the next coordinate
                prediction = model.predict(my_input)

                # add the last predicted coordinate to the list of predictions
                lookahead_predictions.append(prediction[0].tolist())

            # adding the last prediction made to the list of coordinates used for the plotting
            Yhat.append(prediction[0].tolist())

            # adding the real coordinate value to the list of all real coordinates, used to calculate mse
            all_real.append(y_test[i+prediction_steps-1].tolist())
            # adding the predicted coordinate value to the list of all predicted coordinates, used to calculate mse
            all_predicted.append(prediction[0].tolist())

        # plotting both the real and predicted trajectories
        Yhat = np.array(Yhat)
        if plot_data:
            print(Yhat)
            print(y_test)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.plot(Yhat[:, 0], Yhat[:, 1], color='red')
            plt.plot(y_test[:, 0], y_test[:, 1], color='green')
            plt.show()

    # returning the mse beteen the predicted coordinates and the real one
    return mean_squared_error(all_real,all_predicted)


if __name__ == '__main__':
    ########################### IMPORTING DATA #########################
    # reading the dataset file
    dataset = pd.read_csv("new_trajectories.csv")

    # saving the column names
    header = dataset.columns.values

    ########################### SCALING DATA #########################
    from sklearn.preprocessing import MinMaxScaler
    # defining the scaler to map the data in the range (0,1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    # selecting the columns that need to be scaled
    cols_to_scale = ["X (pixel)", "Y (pixel)"]
    # retrieving the selected colums from the dataset
    features = dataset[cols_to_scale]
    # rescaling the retrieved columns
    features = scaler.fit_transform(features)

    # making a copy of the dataset
    dataset_scaled = dataset.copy()
    # inserting in the dataset copy the scaled features
    dataset_scaled[cols_to_scale] = features
    # converting the dataframe into a np array
    dataset_nparray = dataset_scaled.values

    ########################### SPLITTING DATASET #########################
    # splitting the dataset into training and test sets
    training_set, test_set = split_dataset(dataset_nparray, test_size = 0.2, random_state = 0)

    # selecting the training size (how many coordinates in sequence are used to predict the next one)
    look_back = 3

    # defining the model
    model = define_model(look_back)

    # training the model (or loading the weights of a previously trained one)
    model.load_weights("model3.h5")
    #train_model(model, training_set, look_back, save_model=True, filename="model3.h5")

    ########################### TESTING MODEL PERFORMANCE #########################
    # inizializing the list of mses for the different prediction_steps values
    mses =[]
    maxsteps = 10
    for steps in range(1,maxsteps):
        print("Evaluating model on " + str(steps) + " prediction steps")
        mse = evaluate_model(model,test_set, look_back, prediction_steps=steps, plot_data=True)
        print("Mse: " + str(mse))
        mses.append(mse)

    plt.plot(range(1,maxsteps),mses,color='red')
    plt.show()




