import numpy as np

import matplotlib.pyplot as plt 
# I am gonna use: figure, plot, legend, subplot, hist, xlabel, ylim, show, title
from scipy.linalg import svd
import pandas as pd
import sys
from scipy.stats import zscore # to standardize the matrix "samples x features"
# scipy.stats.zscore(a, axis=0, ddof=0, nan_policy='propagate')

def dataframe_setup():
    """
    Import data as dataframe
    convert units of measure, write units of measure
    add BMI
    export tables: dataframe and summary statistics
    """

    # select path to dataset (csv file)

    filename = ('../bodyfat_dataset.txt')

    # create dataframe object
    pd_dataframe = pd.read_csv(filename, delimiter="\t")

    # Adding BMI to the data frame
    # pd_dataframe = pd_dataframe.assign(BMI = pd.Series(np.zeros(len(pd_dataframe))))
    # pd_dataframe['BMI'] = pd_dataframe['Weight'].div(pd_dataframe['Height'].values,axis=0).div(pd_dataframe['Height'].values,axis=0)
    # pd_dataframe['BMI'] = pd_dataframe['BMI'].multiply(703,axis='index')
    
    # convert weight and height to decent units of measure: kg and cm
    pd_dataframe.Weight = pd_dataframe.Weight/2.205
    pd_dataframe.Height = pd_dataframe.Height*2.54

    # this dictionary is used to rename the columns of the dataframe
    # each key has "" as value
    new_columns = {}
    M = len(pd_dataframe.columns.values) # number of columns
    
    new_columns[pd_dataframe.columns.values[0]] = pd_dataframe.columns.values[0]+" [$kg/m^3$]"
    new_columns[pd_dataframe.columns.values[1]]  = pd_dataframe.columns.values[1] + " [%]"
    new_columns[pd_dataframe.columns.values[2]]  = pd_dataframe.columns.values[2] + " [years]"
    new_columns[pd_dataframe.columns.values[3]] =  pd_dataframe.columns.values[3] + " [kg]"
    # from element 4 to elemnt 15
    for i in range(4,16):
        new_columns[pd_dataframe.columns.values[i]] =  pd_dataframe.columns.values[i] + " [cm]"
    
    # # add bmi unit
    # new_columns[pd_dataframe.columns.values[16]] =  pd_dataframe.columns.values[16] + " [$kg/m^2$]"

    # add units of measure
    pd_dataframe = pd_dataframe.rename(columns=new_columns)

    # delete sample with impossible PBF
    for i in range(len(pd_dataframe["PBF [%]"])):
        if pd_dataframe["PBF [%]"][i] < 2:
            pd_dataframe = pd_dataframe.drop(index=i)
            print(i)
    # save as csv
    pd_dataframe.to_csv("../data/modified_dataset/modified_bodyfat_dataset.txt")


    print("opened dataframe\nconverted units of measure\n")
    return pd_dataframe


pd_dataframe = dataframe_setup()
#pd_dataframe = pd_dataframe.reindex(columns=np.flip(pd_dataframe.columns.values))
def plot_histograms(pd_dataframe,save=False):
    """
    INPUT: dataframe, save (if True, will save histograms figures)
    - plots histograms
    - saves figures
    OUTPUT: nothing 
    """

    M = len(pd_dataframe.columns.values) # number of columns
    for i in range(M):
        
        fig = plt.figure(i)
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111)

        # define the plot: histogram of the samples of the i-th colums
        # use the i-th column of data file
        ax.hist(pd_dataframe[pd_dataframe.columns.values[i]],bins=50)
        # no need for axes labels, legend, and grid, I guess? just the title
        ax.set_title(pd_dataframe.columns.values[i]) # NOTE: in this case, it's a subplot!

        # select True as parameter to save the histograms
        # NOTE: use only first word of column label to save. units of measure interfere with the path!
        if save==True:
            fig.savefig("../data/histograms/"+pd_dataframe.columns.values[i].split(" ")[0]+".png")
    
    print("all histograms plotted. returned 0")
    return 0

plot_histograms(pd_dataframe,False)