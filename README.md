## Data Mining - Linear Regression
Luisa Rosa - Spring 2024
---
## Instructions:
+ Download all files (2 Python programs and 6 CSV datasets)
+ Run main.py
+ Save plots if needed
+ To see the answers to the questions, they will printed out to the terminal but are also included here and in the pdf

---
## Question 1: 
Implement L2 regularized linear regression algorithm with λ ranging from 0 to 150(integers only). For each of the 6 datasets, plot both the training set MSE and the test set MSE as a function of λ (x-axis) in one graph.

    * (a) For each dataset, which λ value gives the least test set MSE?

    * (b) For each of the datasets 100-100, 50(1000)-100, 100(1000)-100, provide an additional graph with λ ranging from 1 to 150.

    * (c) Explain why λ = 0 (i.e., no regularization) gives abnormally large MSEs for those three datasets in (b).

### Solution:
Linear Regression (supervised learning) with L2 Regression.
Given N observations, a regression problem tries to accurately predict the corresponding value y = f(x), to each new input value x.

1) - We want to get two matrices that need to be the same size to perform any operations.
x = (x1, x2, ... , xd)
y = (w0 + w1x1 + w2x2 + ... + wdxd) = xw
w is a transpsed vector (w0, w1, ..., wd), and w is the parameter to estimate.
Therefore, the first step is to convert all data frames to matrices, prepend 1 to the matrix data x (add a column of 1's), extract x = (x1, x2, ... , xd), and extract y column.

2) - L2 is a closed-form solution
w = (xTx + lambdaI)^-1 xTy
We need to calculate the weight of the linear regression (lambda) ranging from 1 to 150.

3) - Mean Squared Error (MSE) finds the value that minimizes the errors.
E(w) = 1/n ||Xw-y||^2
take the difference of the predicted and actual values and square the values, then take the average.

4) - The derivative of E(w) = 0 gives us the minimizing error measure. This will allow us to choose the lambda with the least test set MSE.

5) - Then, we're ready to plot the train and test MSE for lambda ranging from 0 to 150 (for each dataset).

---

a) For the test dataset A lambda 9 gives the least MSE 4.1596639277780625
For the test dataset B lambda 22 gives the least MSE 5.072750457735282
For the test dataset C lambda 27 gives the least MSE 4.318370456639974
For the test dataset D (lambda: 0-150) lambda 8 gives the least MSE 5.512273909883558
For the test dataset E (lambda: 0-150) lambda 19 gives the least MSE 5.196199710503634
For the test dataset F (lambda: 0-150) lambda 24 gives the least MSE 4.843720381414155

b) For the test dataset D (lambda: 1-150) lambda 7 gives the least MSE 5.512273909883558
For the test dataset E (lambda: 1-150) lambda 18 gives the least MSE 5.196199710503634
For the test dataset B (lambda: 1-150) lambda 21 gives the least MSE 5.072750457735282

c) When λ = 0 (no regularization), the model becomes prone to overfitting. Without regularization, the model tries to fit the training data as closely as possible, often taking into account noise and outliers, which can lead to poor generalization on unseen data. This overfitting phenomenon results in abnormally large mean squared errors (MSEs) because the model's predictions deviate significantly from the true values in the test set.

---

## Question 2: 
From the plots in question 1, we can tell which value of λ is best for each dataset once we know the test data and its labels. This is not realistic in real-world applications. In this part, we use cross-validation (CV) to set the value for λ. Implement the 10-fold CV technique discussed in class (pseudo code given in Appendix A) to select the best λ value from the training set.
* (a) Using CV technique, what is the best choice of λ value and the corresponding test set MSE for each of the six datasets?
* (b) How do the values for λ and MSE obtained from CV compare to the choice of λ and MSE in question 1(a)?
* (c) What are the drawbacks of CV?
* (d) What are the factors affecting the performance of CV?

### Solution:
a) For the test dataset A lambda 13 gives the least MSE 4.186549495447378
For the test dataset B lambda 20 gives the least MSE 4.466572219197872
For the test dataset C lambda 39 gives the least MSE 4.139641074529679
For the test dataset D lambda 24 gives the least MSE 5.285221355859347
For the test dataset E lambda 31 gives the least MSE 4.852209825819767
For the test dataset F lambda 47 gives the least MSE 4.876912890852046

b) The lambda and MSE values we generated on question 1(a) differed with the use of cross-validation. In question 1, we will know which value of λ is best for each dataset once we know the test data and its labels. However, with a 10-fold CV we are finding the best λ value from the training set. CV provides a more robust and data-driven approach to selecting the optimal lambda value. It takes into account the dataset's characteristics and helps avoid overfitting. In question 1, lambda varied from 9 to 27, while in question 2, lambda varied from 13 to 47.

c) Cross-Validation increases computational cost and time, as it requires training and testing the model k times. Cost of Computation = K folds x choices of lambda.

d) The performance of CV can be affected by several factors, including the choice of CV technique (k-fold, leave-one-out), the size of the dataset, the variability and complexity of the model being evaluated, and the presence of error or missing data (noise).

---

## Question 3: 
Fix λ = 1, 25, 150. For each of these values, plot a learning curve for the algorithm using the dataset 1000-100.csv. Note: a learning curve plots the performance (i.e., test set MSE) as a function of the size of the training set. To produce the curve, you need to draw random subsets (of increasing sizes) and record performance (MSE) on the corresponding test set when training on these subsets. In order to get smooth curves, you should repeat the process at least 10 times and average the results.

--> LC(x_mtx_train_C, y_mtx_train_C, x_mtx_test_C, y_mtx_test_C, fixed_lambda, 10, 1000, 50)
