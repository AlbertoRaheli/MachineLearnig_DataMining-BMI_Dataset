# exercise 8.1.1

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid,savefig,axvline, hist)
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate
import pandas as pd 
from scipy.stats import zscore

def remove_from_matrix(X,i,column=True):
    """
    INPUT
        array of shape (m,n)
        index of column or row to separate
    OUTPUT
        y : removed element
        matrix: same matrix without the column/row
    
    NOTE: not yet implemented the row cutting but whatever
    """
    # isolate the target
    y = X[:,i]
    # create an index mask for the matrix
    X_cols = list(range(0,i)) + list(range(i+1,X.shape[1]))
    # cut and reassemble the original ma
    X = X[:,X_cols]
    return y,X

data = pd.read_csv("../dataset/modified_bodyfat_dataset.txt", sep="\t") # with BMI

# isolate target column as label array
target_index = list(data.columns).index("PBF [%]")
y,X = remove_from_matrix(data.values,target_index)
y.shape,X.shape
# update names
newnames = list( data.columns[:target_index] ) + list( data.columns[target_index+1:] )
attributeNames = [name for name in newnames]


# TEST:REMOVE density
target_index = attributeNames.index("Density [$kg/m^3$]")
y1,X = remove_from_matrix(X,target_index)
newnames = list( attributeNames[:target_index] ) + list( attributeNames[target_index+1:] )
attributeNames = [name for name in newnames]

# TEST:REMOVE bmi
target_index = attributeNames.index("BMI [$kg/m^2$]")
y2,X = remove_from_matrix(X,target_index)
y.shape,X.shape
newnames = list( attributeNames[:target_index] ) + list( attributeNames[target_index+1:] )
attributeNames = [name for name in newnames]



'''
STANDARDIZE DATA
'''
X = zscore(X,axis=0)
y = zscore(y, axis=0)


N, M = X.shape # number of samples, number of features


# Add offset attribute
X = np.concatenate(( np.ones( (N,1) ) , X), axis=1)
attributeNames = [u'Offset']+attributeNames
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 5
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda: 15 elements, -5 to 9
lambdas = np.power(10.,range(-3,9))

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1)) # rlr regularized linear regression
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1)) # without regularization, I guess?
Error_test_nofeatures = np.empty((K,1))

w_rlr = np.empty((M,K)) # weights, linear regression coefficients
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K)) 

k=0
for train_index, test_index in CV.split(X,y):
    """
    also plots best lambda and effect on lin regr coefficients
    """
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    
    # optimal lambda selection
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # BASELINE: Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # LIN REG WITH REG: Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

    # LIN REGR WITHOUT REGULARIZATION Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
    #m = lm.LinearRegression().fit(X_train, y_train)
    #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
    #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

    # Display the results for the LAST cross-validation fold
    if k == K-1:
        figure(k, figsize=(30,15))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor ($\lambda$)', fontsize=30)
        ylabel('Mean Coefficient Values',fontsize=30)
        axvline(opt_lambda, color="0.15",ls="--" )

        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        legend(attributeNames[1:], loc='best', fontsize=17)
        
        subplot(1,2,2)
        #title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)), fontsize=30)
        semilogx(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        axvline(opt_lambda, color="0.15",ls="--" )
        xlabel('Regularization factor ($\lambda$)',fontsize=30)
        ylabel('Squared error',fontsize=30)
        legend(['Train error','Test error'], fontsize=30)
        grid()
        savefig("../data/lin_regression/regularization.png")

    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    k+=1

show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))
    

fig = plt.figure(figsize=(14,10))
plt.rc('xtick', labelsize=30) 
plt.rc('ytick', labelsize=30) 
ax = fig.add_axes([0,0,1,1])
attributes = attributeNames[1:]
weight = w_rlr[1:,-1]
ax.bar(attributes, weight)
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.savefig("../data/lin_regression/attributes_weight.png")
plt.show()

