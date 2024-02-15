# Luisa Rosa
# HW 1 - Data Mining
# 07/02/2024

from LRfunctions import *

# save data sets as df
# A = 100_10
df_train_A = pd.read_csv("train-100-10.csv")
df_test_A = pd.read_csv("test-100-10.csv")
# B = 100-100
df_train_B = pd.read_csv("train-100-100.csv")
df_test_B = pd.read_csv("test-100-100.csv")
# C = 1000_100
df_train_C = pd.read_csv("train-1000-100.csv")
df_test_C = pd.read_csv("test-1000-100.csv")

# generate 3 new dataframes
# test set for these would be df_test_C
df_train_D = df_train_C.head(50)  # D = 50(1000)_100
df_train_E = df_train_C.head(100)  # E = 100(1000)_100
df_train_F = df_train_C.head(150)  # F = 150(1000)_100

# save cleaned files into dataframe
df_train_D.to_csv("train-50(1000)-100.csv", index=False)
df_train_E.to_csv("train-100(1000)-100.csv", index=False)
df_train_F.to_csv("train-150(1000)-100.csv", index=False)

#data cleaning 
# df_train_A has 2 extra columns: x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,y,,
# print(df_train_A)
df_train_A = df_train_A.drop(df_train_A.columns[-1], axis=1) #x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,y, 
df_train_A = df_train_A.drop(df_train_A.columns[-1], axis=1) #x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,y
# print(df_train_A)

##QUESTION 1
# step 1: convert df to matrix, add 1's column, get x's and y.
# convert each training df, saving the respective variables
x_mtx_train_A, y_mtx_train_A = df_to_matrix(df_train_A)
x_mtx_train_B, y_mtx_train_B = df_to_matrix(df_train_B)
x_mtx_train_C, y_mtx_train_C = df_to_matrix(df_train_C)
x_mtx_train_D, y_mtx_train_D = df_to_matrix(df_train_D)
x_mtx_train_E, y_mtx_train_E = df_to_matrix(df_train_E)
x_mtx_train_F, y_mtx_train_F = df_to_matrix(df_train_F)

# convert each testing df, saving the respective variables
x_mtx_test_A, y_mtx_test_A = df_to_matrix(df_test_A)
x_mtx_test_B, y_mtx_test_B = df_to_matrix(df_test_B)
x_mtx_test_C, y_mtx_test_C = df_to_matrix(df_test_C)

# step 2: perform L2 regression for each dataset
# calculate L2 for each training dataset
w_train_ds_A = L2(x_mtx_train_A, y_mtx_train_A, 0, 150)
w_train_ds_B = L2(x_mtx_train_B, y_mtx_train_B, 0, 150)
w_train_ds_C = L2(x_mtx_train_C, y_mtx_train_C, 0, 150)
w_train_ds_D_0 = L2(x_mtx_train_D, y_mtx_train_D, 0, 150)
w_train_ds_E_0 = L2(x_mtx_train_E, y_mtx_train_E, 0, 150)
w_train_ds_F_0 = L2(x_mtx_train_F, y_mtx_train_F, 0, 150)

# calculate L2 for each testing dataset
w_test_ds_A = L2(x_mtx_test_A, y_mtx_test_A, 0, 150)
w_test_ds_B = L2(x_mtx_test_B, y_mtx_test_B, 0, 150)
w_test_ds_C = L2(x_mtx_test_C, y_mtx_test_C, 0, 150)


# step 3: find Mean Squared Error
# calculate the MSE for each training dataset
MSE_train_ds_A = MSE(x_mtx_train_A, y_mtx_train_A, w_train_ds_A)
MSE_train_ds_B = MSE(x_mtx_train_B, y_mtx_train_B, w_train_ds_B)
MSE_train_ds_C = MSE(x_mtx_train_C, y_mtx_train_C, w_train_ds_C)
MSE_train_ds_D_0 = MSE(x_mtx_train_D, y_mtx_train_D, w_train_ds_D_0)
MSE_train_ds_E_0 = MSE(x_mtx_train_E, y_mtx_train_E, w_train_ds_E_0)
MSE_train_ds_F_0 = MSE(x_mtx_train_F, y_mtx_train_F, w_train_ds_F_0)

# calculate the MSE for each testing dataset
MSE_test_ds_A = MSE(x_mtx_test_A, y_mtx_test_A, w_train_ds_A)
MSE_test_ds_B = MSE(x_mtx_test_B, y_mtx_test_B, w_train_ds_B)
MSE_test_ds_C = MSE(x_mtx_test_C, y_mtx_test_C, w_train_ds_C)
MSE_test_ds_D_0 = MSE(x_mtx_test_C, y_mtx_test_C, w_train_ds_D_0)
MSE_test_ds_E_0 = MSE(x_mtx_test_C, y_mtx_test_C, w_train_ds_E_0)
MSE_test_ds_F_0 = MSE(x_mtx_test_C, y_mtx_test_C, w_train_ds_F_0)


# For each of the 6 datasets(lambda = [0-150]), plot training and testing set MSE as a function of λ.
# dataset A = 100-10
MSE_train_A_plot, MSE_test_A_plot = plot_MSE(
    MSE_train_ds_A, MSE_test_ds_A, "A", "100-10", "[0-150]"
)
# dataset B = 100-100
MSE_train_B_plot, MSE_test_B_plot = plot_MSE(
    MSE_train_ds_B, MSE_test_ds_B, "B", "100-100", "[0-150]"
)
# dataset C = 1000_100
MSE_train_C_plot, MSE_test_C_plot = plot_MSE(
    MSE_train_ds_C, MSE_test_ds_C, "C", "1000-100", "[0-150]"
)
# dataset D = 50(1000)_100
MSE_train_D_0_plot, MSE_test_C_plot = plot_MSE(
    MSE_train_ds_D_0, MSE_test_ds_C, "D", "50(1000)_100", "[0-150]"
)
# dataset E = 100(1000)_100
MSE_train_E_0_plot, MSE_test_C_plot = plot_MSE(
    MSE_train_ds_E_0, MSE_test_ds_C, "E", "100(1000)_100", "[0-150]"
)
# dataset F = 150(1000)_100
MSE_train_F_0_plot, MSE_test_C_plot = plot_MSE(
    MSE_train_ds_F_0, MSE_test_ds_C, "F", "150(1000)_100", "[0-150]"
)


##QUESTION 1(A): For each dataset, which lambda gives the least test set MSE?
print("Question 1a: \n")

# calculate the least MSE for each test set
Least_Test_Set("A", MSE_test_ds_A)
Least_Test_Set("B", MSE_test_ds_B)
Least_Test_Set("C", MSE_test_ds_C)
Least_Test_Set("D (lambda: 0-150)", MSE_test_ds_D_0)
Least_Test_Set("E (lambda: 0-150)", MSE_test_ds_E_0)
Least_Test_Set("F (lambda: 0-150)", MSE_test_ds_F_0)


##QUESTION 1(B): For each of datasets 100-100, 50(1000)-100, 100(1000)-100, provide an additional graph with λ ranging from 1 to 150
print("Question 1b: \n")

# additional weights (L2) calculations with lambda 1-150, train & test
w_train_ds_D_1 = L2(x_mtx_train_D, y_mtx_train_D, 1, 150)
w_train_ds_E_1 = L2(x_mtx_train_E, y_mtx_train_E, 1, 150)
w_train_ds_B_1 = L2(x_mtx_train_B, y_mtx_train_B, 1, 150)

# calculate the MSE for each additional training dataset
MSE_train_ds_D_1 = MSE(x_mtx_train_D, y_mtx_train_D, w_train_ds_D_1)
MSE_train_ds_E_1 = MSE(x_mtx_train_E, y_mtx_train_E, w_train_ds_E_1)
MSE_train_ds_B_1 = MSE(x_mtx_train_B, y_mtx_train_B, w_train_ds_B_1)

# calculate the MSE for each additional testing dataset
MSE_test_ds_D_1 = MSE(x_mtx_test_C, y_mtx_test_C, w_train_ds_D_1)
MSE_test_ds_E_1 = MSE(x_mtx_test_C, y_mtx_test_C, w_train_ds_E_1)
MSE_test_ds_B_1 = MSE(x_mtx_test_B, y_mtx_test_B, w_train_ds_B_1)

# For each of the additional ds(lambda = [1-150]), plot both the training set MSE and the test set MSE
# dataset D = 50(1000)_100
MSE_train_D_1_plot, MSE_test_C_plot = plot_MSE(
    MSE_train_ds_D_1, MSE_test_ds_C, "D", "50(1000)_100", "[1-150]"
)
# dataset E = 100(1000)_100
MSE_train_E_1_plot, MSE_test_C_plot = plot_MSE(
    MSE_train_ds_E_1, MSE_test_ds_C, "E", "100(1000)_100", "[1-150]"
)
# dataset F = 150(1000)_100
MSE_train_F_1_plot, MSE_test_C_plot = plot_MSE(
    MSE_train_ds_B_1, MSE_test_ds_B, "B", "100_100", "[1-150]"
)

# calculate the least MSE for each additional test set
D_1_minl = Least_Test_Set("D (lambda: 1-150)", MSE_test_ds_D_1)
E_1_minl = Least_Test_Set("E (lambda: 1-150)", MSE_test_ds_E_1)
F_1_minl = Least_Test_Set("B (lambda: 1-150)", MSE_test_ds_B_1)


##QUESTION 1(C): Explain why λ = 0 (i.e., no regularization) gives abnormally large MSEs for those three datasets in (b).
print(
    "Question 1c: \n When λ = 0 (no regularization), the model becomes prone to overfitting. Without regularization, the model tries to fit the training data as closely as possible, often taking into account noise and outliers, which can lead to poor generalization on unseen data. This overfitting phenomenon results in abnormally large mean squared errors (MSEs) because the model's predictions deviate significantly from the true values in the test set.\n"
)


##QUESTION 2: Implement the 10-fold CV technique discussed in class to select the best λ value from the training set.
##QUESTION 2(A): Using CV technique, what is the best choice of λ value and the corresponding test set MSE for each of the six datasets? Answer outputted by the Least_Test_Set() function called within the CV function.
print("Question 2a: \n")

A_bestl = CV("A", 10, y_mtx_train_A, x_mtx_train_A, 0, 150)
B_bestl = CV("B", 10, y_mtx_train_B, x_mtx_train_B, 0, 150)
C_bestl = CV("C", 10, y_mtx_train_C, x_mtx_train_C, 0, 150)
D_bestl = CV("D", 10, y_mtx_train_D, x_mtx_train_D, 0, 150)
E_bestl = CV("E", 10, y_mtx_train_E, x_mtx_train_E, 0, 150)
F_bestl = CV("F", 10, y_mtx_train_F, x_mtx_train_F, 0, 150)


##QUESTION 2(B): How do the values for λ and MSE obtained from CV compare to the choice of λ and MSE in question 1(a)?
print("Question 2b:\nThe lambda and MSE values we generated on question 1(a) differed with the use of cross-validation. In question 1, we will know which value of λ is best for each dataset once we know the test data and its labels. However, with a 10 fold CV we are finding the best λ value from the training set. CV provides a more robust and data-driven approach to selecting the optimal lambda value. It takes into account the dataset's characteristics and helps avoid overfitting.In question 1, lambda varied from 9 to 27, while in question 2, lambda varied from 13 to 47.\n")


##QUESTION 2(C): What are the drawbacks of CV?
print("Question 2c:\nCross Validation increases computational cost and time, as it requires training and testing the model k times. Cost of Computation = K folds x choices of lambda.\n") 


##QUESTION 2(D): What are the factors affecting the performance of CV?
print("Question 2d:\nThe performance of CV can be affected by several factors, including the choice of CV technique (k-fold, leave-one-out), the size of the dataset, the variability and complexity of the model being evaluated, and the presence of error or missing data (noise).\n")


##QUESTION 3: Plot a learning curve for the algorithmusing the dataset 1000-100.csv fixing lambda = 1, 25, 150.

# Learning Curve Function: plots the test set MSE as a function of the size of the training set. To produce the curve, you need to draw random subsets (of increasing sizes) and record performance on the corresponding test set when training on these subsets. In order to get smooth curves, you should repeat the process at least 10 times and averagethe results.

fixed_lambda = [1, 25, 150]

LC(x_mtx_train_C, y_mtx_train_C, x_mtx_test_C, y_mtx_test_C, fixed_lambda, 10, 1000, 50)