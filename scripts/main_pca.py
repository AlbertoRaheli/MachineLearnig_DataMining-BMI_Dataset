
import numpy as np

import matplotlib.pyplot as plt 
# I am gonna use: figure, plot, legend, subplot, hist, xlabel, ylim, show, title
from scipy.linalg import svd
import pandas as pd
from scipy.stats import zscore # to standardize the matrix "samples x features"
# scipy.stats.zscore(a, axis=0, ddof=0, nan_policy='propagate')

def dataframe_setup():
    """
    Import data as dataframe
    convert units of measure, write units of measure
    add BMI
    export tables: dataframe and summary statistics
    """

    import numpy as np

    import matplotlib.pyplot as plt 
    # I am gonna use: figure, plot, legend, subplot, hist, xlabel, ylim, show, title

    import pandas as pd

    # select path to dataset (csv file)
    filename='../Data/bodyfat_dataset.txt'

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

    import numpy as np

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
            fig.savefig("../Data/histograms/"+pd_dataframe.columns.values[i].split(" ")[0]+".png")
    
    print("all histograms plotted. returned 0")
    return 0

#plot_histograms(pd_dataframe,True)

def summary_statistics(pd_dataframe):
    """
    calculate summary statistics and export csv table
    """

    # create a dataframe with summary statistics     
    summary_stat = pd_dataframe.describe()

    # print it?
    # print(summary_stat)

    # export comma separated value
    path = '../Data/summary_statistics_table.txt'
    summary_stat.to_csv(path)

    return summary_stat

#summary_stat = summary_statistics(pd_dataframe)


def correlation_heatmap(pd_dataframe):
    from scipy.stats import zscore
    data = zscore(pd_dataframe,axis=0,ddof=1) # this returns a numpy ndarray, matrix of dimension sample x parameters
    # https://numpy.org/doc/stable/reference/generated/numpy.cov.html
    
    # correlation coefficient matrix
    corr = np.corrcoef(data,rowvar=False,ddof=1)

    fig = plt.figure(1,figsize=(25,35))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    colours = ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper']

    heatmap = ax.imshow(corr, cmap=colours[13], interpolation='nearest')

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[1]))
    ax.set_xticklabels (pd_dataframe.columns.values, fontsize=30, rotation=90)
    ax.set_yticklabels (pd_dataframe.columns.values,fontsize=30)
    ax.xaxis.tick_top()
    
    # add the colorbar using the figure's method, telling which mappable we're talking about and which axes object it should be near
    col_bar = fig.colorbar(heatmap, ax=ax,shrink=0.6)
    col_bar.ax.tick_params(labelsize=30)    
    fig.savefig("../Data/heatmaps/heatmap.png")

#correlation_heatmap(pd_dataframe)

#pd_dataframe_noPBF = pd_dataframe.drop(columns=['PBF [%]'])
def scatterplot(pd_dataframe,save=True):
    """
    scatterplot all parameters
    INPUT:
        dataframe
        indexes of the columns
    """
    col_num = len(pd_dataframe.columns.values)
    k=0
    for i in range(col_num):
        for j in range(i,col_num):
            k+=1
            fig = plt.figure(k)
            fig.patch.set_facecolor("white")
            ax = fig.add_subplot(111)
            ax.scatter(pd_dataframe[pd_dataframe.columns.values[i]],pd_dataframe[pd_dataframe.columns.values[j]])

            ax.set_xlabel(pd_dataframe.columns.values[i]) 
            ax.set_ylabel(pd_dataframe.columns.values[j])
            fig.savefig("../data/scatterplots/"+str(k)+"_"+pd_dataframe.columns.values[i].split(" ")[0]+"_vs_"+pd_dataframe.columns.values[j].split(" ")[0])
    return 0

scatterplot(pd_dataframe)

def raw_data(pd_dataframe):
    # Creating raw data with one out of K encoding
    data = np.array(pd_dataframe.values)
    #data = np.array(pd_dataframe.values)
    N, M = data.shape
    temp = np.zeros((N,1))
    data = np.concatenate((data,temp),axis=1)
    #assigning classes accordingly to the BF%
    for it1 in range(len(data)):
        col1 = 1 #column that we use for classification
        if data[it1,col1] < 14:
            data[it1,-1] = 0    # athletes
        if data[it1,col1] >= 14 and data[it1,1] < 18:
            data[it1,-1] = 1    # fitness
        if data[it1,col1] >= 18 and data[it1,1] < 24:
            data[it1,-1] = 2   # average
        if data[it1,col1] >= 24:
            data[it1,-1] = 3  #obese
    # One out of K encoding
    categories = np.array(data[:, -1], dtype=int).T    
    #categories = np.resize(categories,(len(categories),1))
    K = categories.max()+1
    categories_encoding = np.zeros((categories.size, K))
    categories_encoding[np.arange(categories.size), categories] = 1
    data = np.concatenate( (data[:, :-1], categories_encoding), axis=1) 
    
    attribute_names = pd_dataframe.columns.values
    # adding names for encoded parameters
    classNames = ['Athletic', 'Fitness','Average','Obese']
    #C = len(classNames)
    # attribute_names = np.concatenate((attribute_names, classNames),axis=0)
    attribute_names = np.delete(attribute_names,1)
    attribute_names = np.resize(attribute_names,(len(attribute_names),1))
    
    data = np.delete(data, 1, axis=1)
    data = np.delete(data, np.s_[15:19],axis=1)
    print("pd_dataframe converted to raw data\n")

    return data, attribute_names, classNames, categories
    
data, attribute_names, classNames, categories = raw_data(pd_dataframe)



def pca_fun(data,i,j,classNames, categories):
    
    N,M = data.shape
    #with standard deviation
    Y = (data - np.ones((N,1))*data.mean(axis=0))/np.std(data)
    
    # PCA by computing SVD of Y
    U,S,Vh = svd(Y,full_matrices=False)
    # scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
    # of the vector V. So, for us to obtain the correct V, we transpose:
    V = Vh.T 
    
    # Project the centered data onto principal component space
    Z = Y @ V 
    
    # Compute variance explained by principal components
    rho = (S*S) / (S*S).sum() 
    
    threshold95 = 0.95
    
    # Plot variance explained
    plt.figure()
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.plot([1,len(rho)],[threshold95, threshold95],'k--')
    plt.title('Variance explained by principal components');
    plt.xlabel('Principal component');
    plt.ylabel('Variance explained');
    plt.legend(['Individual','Cumulative','Threshold'])
    plt.grid()
    plt.show()

    # Indices of the principal components to be plotted
    #i = 0
    #j = 20
    
    # Plot PCA of the data
    plt.figure()
    plt.title('Bodyfat: PC{0} '.format(i+1) + 'vs PC{0}'.format(j+1))
    
    C = len(classNames)
    
    for c in range(C):
        # select indices belonging to class c:
        class_mask = categories==c
        if c == 0:
            plt.plot(Z[class_mask,i], Z[class_mask,j],'or', alpha=0.8)
        if c == 1:
            plt.plot(Z[class_mask,i], Z[class_mask,j],'ok', alpha=0.7)
        if c == 2:
            plt.plot(Z[class_mask,i], Z[class_mask,j],'og', alpha=0.8)
        if c == 3:
            plt.plot(Z[class_mask,i], Z[class_mask,j],'ob', alpha=0.8)
    plt.legend(classNames)
    plt.xlabel('PC{0}'.format(i+1))
    plt.ylabel('PC{0}'.format(j+1))
    
    # Output result to screen
    plt.show()

    # Components that explain more than 95 percent of the variance.
    # Let's look at their coefficients:
    plt.figure()
    pcs = [0,1,2,3]
    legendStrs = ['PC'+str(e+1) for e in pcs]
    c = ['r','k','b','y']
    bw = .2
    r = np.arange(1,M+1)
    for i in pcs:    
        bar1 = plt.bar(r+i*bw, V[:,i], width=bw)
        
    # PLOTTING 

    plt.xticks(r+bw, attribute_names)
    plt.xticks(rotation=90)
    #bar1.xaxis.set_major_locator(plt.ticker.FixedLocator([1,5,8])) 
    plt.xlabel('Attributes')
    plt.ylabel('Component coefficients')
    plt.legend(legendStrs)
    plt.grid()
    plt.title('Body fat: PCA Component Coefficients')
    plt.show()
    print('PCA done')
    
i=0
j=1
pca_fun(data,i,j,classNames,categories)    
    