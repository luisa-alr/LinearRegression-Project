# Luisa Rosa
# HW 1 - Data Mining
# 07/02/2024

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Function that converts df to matrix, add 1's column, and get x's and y's matrices.
def df_to_matrix(df):
    df.insert(0, "x0", 1)
    xs = df.iloc[:, :-1] #get all x's columns
    x_matrix = xs.to_numpy()
    y = df.iloc[:,-1:] #get y column
    y_matrix = y.to_numpy()

    return x_matrix, y_matrix

# Function to calculate L2 regression
def L2(x_mtx, y_mtx, start, l_range):
    l_values = np.arange(start, l_range + 1)
    w_corresponding_l = []  # list to store diff values of lambda

    for l in l_values:
        # w = (xTx + lambdaI)^-1 xTy
        x_mtx_T = np.transpose(x_mtx)  # transpose x matrix
        xTx = np.dot(x_mtx_T, x_mtx)  # multiply matrices
        I = np.identity(len(xTx))  # create identity matrix
        lambdaI = np.dot(l, I)  # lambda * identity matrix
        parenthesis = xTx + lambdaI  # sum
        inverse = np.linalg.inv(parenthesis)  # take inverse (^-1)
        xTy = np.dot(x_mtx_T, y_mtx)  # multiply matrices
        w = np.dot(inverse, xTy)  # find w

        # save w flatten (reducing the dimensions of the array)
        w_corresponding_l.append(w.flatten())
        w_dataset = np.transpose(np.array(w_corresponding_l))
    return w_dataset

# Function to find Mean Squared Error
def MSE(x_mtx, y_mtx, weights):
    # E(w) = 1/n ||Xw-y||^2
    y_predictions = np.dot(x_mtx, weights)  # multiply matrix X by w
    sum = 0.0
    for n in range(len(y_mtx)):
        pred_diff = y_predictions[n] - y_mtx[n] # for each n, get the prediction - the actual value of y to get the prediction difference
        sum += (pred_diff**2)  # for all n, sum the square of the differences
    E = sum / float(len(y_mtx))
    return E

# Function that plots both the training set MSE and the test set MSE as a function of λ (x-axis) in one graph.
def plot_MSE(MSE_train_ds, MSE_test_ds, ds, size, l):
    MSE_train_plot = plt.plot(
        MSE_train_ds, label=f"MSE Dataset {ds} - Train", color="darkorange"
    )
    MSE_test_plot = plt.plot(
        MSE_test_ds, label=f"MSE Dataset {ds} - Test", color="dodgerblue"
    )
    plt.title(f"Train-{size} vs Test-{size}\n lambda {l}")
    plt.xlabel("lambda")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

    return MSE_train_plot, MSE_test_plot

# Function to find the minimum size of the test set (which lambda gives the min MSE)
def Least_Test_Set(ds, mse):
    min_lambda = mse.argmin()
    min_mse = mse[min_lambda]

    print(f"For the test dataset {ds} lambda {min_lambda} gives the least MSE {min_mse}\n")

#Function to perform Cross Validation to select the best λ value from the training set.
def CV(ds, K_value, y_mtx, x_mtx, start, l_range):
    # split the data into K=10 disjoint folds
    fold_size = int(len(y_mtx) / K_value)

    mse_sum = 0
    for i in range(K_value):
        # train A(lambda) on all folds but the ith fold
        x_train_fold = np.concatenate(
            (x_mtx[: i * fold_size], x_mtx[(i + 1) * fold_size :]), axis=0
        )
        y_train_fold = np.concatenate(
            (y_mtx[: i * fold_size], y_mtx[(i + 1) * fold_size :]), axis=0
        )
        # test on ith fold and record error on fold i
        x_test_fold = x_mtx[i * fold_size : (i + 1) * fold_size]
        y_test_fold = y_mtx[i * fold_size : (i + 1) * fold_size]

        weights = L2(x_train_fold, y_train_fold, start, l_range)
        mse_sum += MSE(x_test_fold, y_test_fold, weights)

    # compute the average performance of lambda in the 10 folds
    mse_test = mse_sum / K_value

    # pick the value of lambda with the best average performance
    bestl = Least_Test_Set(ds, mse_test)

    return bestl

# Function to find and plot the Learning Curve
def LC(x_train, y_train, x_test, y_test, fixed_lambda, rep, max_size, step):
    for l in fixed_lambda:
        sizes = range(10, max_size+1, step)  # Generate sizes from 10 to max_size with a step size of 'step'
        MSE_array_test = np.zeros(len(sizes))
        MSE_array_train = np.zeros(len(sizes))
        for i, size in enumerate(sizes):
            rep_list_test = []
            rep_list_train = []
            for j in range(rep):
                # Randomly select 'size' samples from the training set
                idx = np.random.choice(len(x_train), size, replace=False)
                x_train_subset = x_train[idx]
                y_train_subset = y_train[idx]
                
                # Train the model using the subset of training data
                weight = L2(x_train_subset, y_train_subset, l, l)

                # Calculate MSE on the test set using the trained model
                mse_test = MSE(x_test, y_test, weight)
                mse_train = MSE(x_train_subset, y_train_subset, weight)

                rep_list_test.append(mse_test)
                rep_list_train.append(mse_train)
            # Average the MSE values over 'rep' repetitions
            MSE_array_test[i] = np.average(rep_list_test)
            MSE_array_train[i] = np.average(rep_list_train)
            
        # Get the middle x-coordinate of the plot
        middle_x = (sizes[0] + sizes[-1]) / 2
        # Plot the learning curve for the current lambda value
        plt.plot(sizes, MSE_array_test, label="MSE_test", color='deeppink')
        plt.plot(sizes, MSE_array_train, label="MSE_train", color='blue')
        plt.text(middle_x, MSE_array_test[1], f"E$_{{out}}$", va='bottom', color='deeppink')
        plt.text(middle_x, MSE_array_train[1], f"E$_{{in}}$", va='top', color='blue')
        plt.xlabel("Training Set Size")
        plt.ylabel(f"Expected Error with lambda = {l}")
        plt.legend()
        plt.show()