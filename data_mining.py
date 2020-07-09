import pandas as pd                # Pandas for dealing with DataFrames.
import numpy as np                 # Numpy package.
import matplotlib.pyplot as plt    # Graphs module from matplotlib.
import seaborn as sns              # For plotting more sophisticated graphs.
import random                      # Random package, for setting seeds.
import urllib

from sklearn.cluster import KMeans # kmeans function from sklearn package.
from sklearn.preprocessing import StandardScaler # Scaling function from sklearn package.
from sklearn.decomposition import PCA # For working with PCA.
from sklearn.cluster import AgglomerativeClustering  # For agglomerative clustering.
from sklearn import mixture # For EM Clustering (GaussianMixture).
import sklearn.metrics as sm  # Metrics from sklearn package.
from scipy.cluster.hierarchy import dendrogram, linkage # To work with Hierarchical dendograms.
from collections import Counter # For counting number of elements in each cluster.
from sklearn.cluster import MeanShift, estimate_bandwidth # Mean-Shift Clustering.
from kmodes.kmodes import KModes # needs  to be installed in terminal with comand  : pip install kmodes

url="https://raw.githubusercontent.com/ritmas1/unsupervised_learning_marketing/master/Insurance.csv"
urllib.request.urlretrieve(url,'insurance.csv')  
data=pd.read_csv('insurance.csv')

#data=pd.read_csv("../../Desktop/data/Insurance.csv")

#printing out the column names of our dataset and its type
print(data.dtypes)

#columns with numerical values
data.select_dtypes(exclude=["object","bool_"]).columns
#however this list includes

data.select_dtypes(include=["bool_"]).columns
#no columns here, however, column 'Has Children(Y=1) is obviuosly of a boolean type'
#columns with categorical data
data.select_dtypes(include=["category",'object']).columns


data.iloc[:, [4, 9, 10, 11, 12]].describe()

# 1. Identifying missing values and outliers and investigating correlated features

# 1.1 identifying missing values 
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = df.isnull().sum() / len(df) * 100
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    return mis_val_table_ren_columns

missing_values_table(data)


# 1.2 Outliers
#We should remove outliers, as they are highly prejudicial to K-Means results, since the outliers are going to pull the centroid of the clusters

def plotting_outliers(variable,threshold1,threshold2,axline):
    fig, ax =plt.subplots(1,2)
    sns.boxplot(data[str(variable)].loc[data[str(variable)] <= threshold1].dropna(), orient="v", whis=1.0, ax=ax[0])
    ax[0].axhline(y=axline, color='r')
    sns.boxplot(data[str(variable)].loc[data[str(variable)] <= threshold2].dropna(), orient="v", whis=1.0, ax=ax[1])
    plt.show()
    
#first, let's have a look at First Policy´s Year'      
variable='First Policy´s Year'   
threshold1=90000
threshold2=2020
axline=threshold2
plotting_outliers(variable,threshold1,threshold2,axline)    
# we can see there is one extreme and unrealistic value above year 2020


#gross monthly salary

variable='Gross Monthly Salary'   
threshold1=90000
threshold2=30000
axline=threshold2
plotting_outliers(variable,threshold1,threshold2,axline)


#premiums MOTOR
variable='Premiums in LOB: Motor'   
threshold1=90000
threshold2=3000
axline=threshold2
plotting_outliers(variable,threshold1,threshold2,axline)

#premiums HOUSEHOLD
variable='Premiums in LOB: Household'   
threshold1=90000
threshold2=3000
axline=threshold2
plotting_outliers(variable,threshold1,threshold2,axline)

#premiuns HEALTH
variable='Premiums in LOB: Health'   
threshold1=90000
threshold2=3000
axline=threshold2
plotting_outliers(variable,threshold1,threshold2,axline)

#premiums WORK COMPENSATION
variable='Premiums in LOB: Work Compensations'   
threshold1=90000
threshold2=750
axline=threshold2
plotting_outliers(variable,threshold1,threshold2,axline)

#claims rate
variable='Claims Rate'   
threshold1=90000
threshold2=4
axline=threshold2
plotting_outliers(variable,threshold1,threshold2,axline)



n0 = data.shape[0] # Storing original size of dataset.
data = data[data["First Policy´s Year"]<3000] # Removing 1 extreme value in the first year of policy.
data = data[data["Gross Monthly Salary"]<30000] # Removing the 2 extreme values present in Gross Monthly Salary.
data = data[data["Premiums in LOB: Motor"]<3000] # Removing some extreme values in the Premiums in LOB: Motor.
data = data[data["Premiums in LOB: Household"]<3000] # Removing 3 extreme values in the Premiums in LOB: Household.
data = data[data["Premiums in LOB: Health"]<3000] # Removing 2 extreme values in the Premiums in LOB: Health.
data = data[data["Premiums in LOB: Work Compensations"]<750] # Removing 2 extreme values in the Premiums in LOB: Work Compensations.
data = data[data["Claims Rate"] < 4] # Removing enough extreme values so the distribution of Claims Rate is close enough to a Gaussian one.
n1=data.shape[0]

#1.3 correlation heatmap
# Compute the correlation matrix
corr = data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()


#We can observe that Birthday Year shows a high negative correlation with Gross Monthly Salary. 
#Since the variable Birthday Year shows dicrepancy with the variable First Policys Year, which does not show high correlation with any other variable, we will drop the variable Birthday Year.

print("Correlation between Birthday Year and Gross Monthly Salary:", corr.loc["Brithday Year", "Gross Monthly Salary"])
data.drop(columns=["Brithday Year"], inplace=True)
data.dropna(inplace=True)

#Since the variables Claims Rate and Monetary Value show a really high negative correlation as well, we should also remove one of them in order to reduce dimensionality and preserve orthogonality.
print("Correlation between Claims Rate and Customer Monetary Value:", corr.loc["Claims Rate", "Customer Monetary Value"])
data.drop(columns=["Customer Monetary Value"], inplace=True)
data.dropna(inplace=True)


# 2   CREATING NEW VARIABLES
#2.1.Creating time variable subtracting the first year from 2016

data["Time"] = 2016-data["First Policy´s Year"]
data = data.drop(columns="First Policy´s Year") # Drop the original column.

#2.2 Creating dummy variable for Education:
aux_1 = pd.get_dummies(data["Educational Degree"], dummy_na = True)
data["ed_Basic"]   = aux_1.iloc[:, 0]
data["ed_HS"]      = aux_1.iloc[:, 1]
data["ed_BSc/MSc"] = aux_1.iloc[:, 2]
data["ed_Phd"]     = aux_1.iloc[:, 4]

#2.3 Creating dummy variable for Geographic live area:
    
aux_2 = pd.get_dummies(data["Geographic Living Area"], dummy_na = True)
data["GLA_1"] = aux_2.iloc[:, 0]
data["GLA_2"] = aux_2.iloc[:, 1]
data["GLA_3"] = aux_2.iloc[:, 2]
data["GLA_4"] = aux_2.iloc[:, 4]

#2.4 Creating Percentage of active contracts:
df_Premium = data.loc[:,('Premiums in LOB: Motor','Premiums in LOB: Household','Premiums in LOB: Health','Premiums in LOB:  Life','Premiums in LOB: Work Compensations')]
df_Premium['Premiums in LOB: Motor'] = df_Premium['Premiums in LOB: Motor'].apply(lambda x: 1 if x > 0 else 0)
df_Premium['Premiums in LOB: Household'] = df_Premium['Premiums in LOB: Household'].apply(lambda x: 1 if x > 0 else 0)
df_Premium['Premiums in LOB: Health'] = df_Premium['Premiums in LOB: Health'].apply(lambda x: 1 if x > 0 else 0)
df_Premium['Premiums in LOB:  Life'] = df_Premium['Premiums in LOB:  Life'].apply(lambda x: 1 if x > 0 else 0)
df_Premium['Premiums in LOB: Work Compensations'] = df_Premium['Premiums in LOB: Work Compensations'].apply(lambda x: 1 if x > 0 else 0)
df_Premium["%NumContracts"] = df_Premium.sum(axis=1)/5
df_Premium = df_Premium.drop(columns=['Premiums in LOB: Motor','Premiums in LOB: Household','Premiums in LOB: Health','Premiums in LOB:  Life','Premiums in LOB: Work Compensations'])
data = pd.concat([data, df_Premium], axis=1)
del df_Premium

#2.5.Creating Percentage of spend by contract (might take a little while to run this one):
df_Premium = data.loc[:,('Premiums in LOB: Motor','Premiums in LOB: Household','Premiums in LOB: Health','Premiums in LOB:  Life','Premiums in LOB: Work Compensations')]
df_Premium['Premiums in LOB: Motor'] = df_Premium['Premiums in LOB: Motor'].apply(lambda x: x if x > 0 else 0)
df_Premium['Premiums in LOB: Household'] = df_Premium['Premiums in LOB: Household'].apply(lambda x: x if x > 0 else 0)
df_Premium['Premiums in LOB: Health'] = df_Premium['Premiums in LOB: Health'].apply(lambda x: x if x > 0 else 0)
df_Premium['Premiums in LOB:  Life'] = df_Premium['Premiums in LOB:  Life'].apply(lambda x: x if x > 0 else 0)
df_Premium['Premiums in LOB: Work Compensations'] = df_Premium['Premiums in LOB: Work Compensations'].apply(lambda x: x if x > 0 else 0)
df_Premium['TotalSpent'] = df_Premium.sum(axis=1)

df_Premium['Premiums in LOB: Motor'] = df_Premium['Premiums in LOB: Motor'].apply(lambda x: x/df_Premium['TotalSpent'])
df_Premium['Premiums in LOB: Household'] = df_Premium['Premiums in LOB: Household'].apply(lambda x: x/df_Premium['TotalSpent'])
df_Premium['Premiums in LOB: Health'] = df_Premium['Premiums in LOB: Health'].apply(lambda x: x/df_Premium['TotalSpent'])
df_Premium['Premiums in LOB:  Life'] = df_Premium['Premiums in LOB:  Life'].apply(lambda x: x/df_Premium['TotalSpent'])
df_Premium['Premiums in LOB: Work Compensations'] = df_Premium['Premiums in LOB: Work Compensations'].apply(lambda x: x/df_Premium['TotalSpent'])

data = data.drop(columns=['Premiums in LOB: Motor','Premiums in LOB: Household','Premiums in LOB: Health','Premiums in LOB:  Life','Premiums in LOB: Work Compensations'])
data = pd.concat([data, df_Premium], axis=1)





# 3 - Clustering
#We will use Hierarchical Clustering for selecting number of clusters and K-Means or K-Modes to cluster data.
#Looking at the shape of our data, doesnt look like we need DBSCAN. Since it does not scale well, we should not use it.
#EM Clustering could be a viable option if K-Means cant do the job well enough.
#Mean Shift is another option where we dont need to select the number of Clusters.

# 3.1- Clustering "Customers'Value" (here we only consider two features which proved to show meaningful clusters 
#during several trail attempts: namely "Gross Monthly Salary" and "Claims Rate")

# 3.1.1 - Select the features
def plotting(data,centers_cv=False,labels=None,x=0,y=1):
    plt.figure(num=None, figsize=(7, 7), dpi=80, facecolor='w', edgecolor='k')
    plt.scatter(data.iloc[:, x], data.iloc[:, y], c=labels, s=5, cmap='viridis')
    if centers_cv.all()!=False:
        plt.scatter(centers_cv[:, x], centers_cv[:, y], c='black', s=200, alpha=0.5)
    plt.xlabel(data.columns[x], fontsize=10)
    plt.ylabel(data.columns[y], fontsize=10)
    plt.show()
  

#Lets select Gross Monthly Salary and Claims Rate to try and cluster the clients according to their value to the company.

features_df = data[["Gross Monthly Salary","Claims Rate"]]
# Ploting variables
plotting(data=features_df)      
    
    
# 3.1.2 - Scaling
#K-means clustering is "isotropic" in all directions of space and therefore tends to produce more or less round 
#(rather than elongated) clusters. 
#In this situation leaving variances unequal is equivalent to putting more weight on variables with smaller variance.
# So we scale the variables in order to remove the difference in magnitude of the clustering analysis.

scaled_features = StandardScaler().fit_transform(features_df.values)
scaled_features_df = pd.DataFrame(scaled_features, index=features_df.index, columns=features_df.columns)
scaled_features_df.describe()


# 3.1.3 - Elbow Graph
def elbow_graph(data):
    wcss = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, max_iter=1000, init = 'k-means++', 
                        random_state = 12345)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    # Plot the Elbow Graph
    plt.figure(num=None, figsize=(7, 7), dpi=80, facecolor='w', edgecolor='k')
    
    plt.plot(range(1, 11), wcss)
    plt.plot(range(1, 11), wcss, "rs")
    
    plt.xlabel("Number of Clusters")
    plt.ylabel("Within Cluster Sum of Errors")
    
    plt.show()

elbow_graph(scaled_features_df)
#The Elbow Graph indicates some number between 3 and 5 clusters. Lets see what the Dendogram tells us.

# 3.1.4 - Hierarchical (Dendrogram)

def dendrogram_(data):
    Z = linkage(data,
                method = 'ward')
    
    plt.figure(num=None, figsize=(7, 7), dpi=80, facecolor='w', edgecolor='k')
    
    dendrogram(Z,
               truncate_mode='lastp',
               p=7,
               orientation = 'top',
               leaf_rotation=45.,
               leaf_font_size=10.,
               show_contracted=True,
               show_leaf_counts=True)
    
    plt.title('Truncated Hierarchical Clustering Dendrogram')
    plt.xlabel('Cluster Size')
    plt.ylabel('Distance')
    
    plt.axhline(y=65)
    plt.show()

dendrogram_(scaled_features_df)    

#Even though it is clear that 3 clusters is where we maximize the distance between forks of new clusters,
# for our purposes of separating the clients in 4 quadrants (see next plot) and group them into High-Low Claims Rate and Salary 4 is the best numer of clusters.

# 3.1.5 - K-Means
# Setting up the K-Means model:

kmeans = KMeans(n_clusters=4, max_iter=1000, random_state=12345, init = 'k-means++') # Set the number of clusters we want, the maximum number of iteration and a seed for the random state.
kmeans.fit(scaled_features_df)
labels = kmeans.labels_
centers_cv=kmeans.cluster_centers_

# Plotting the resulting clusters:
plotting(data=scaled_features_df,centers_cv=kmeans.cluster_centers_,labels=labels)              
  

# 3.2 Clustering "Consumption" here we only consider features which proved to show meaningful clusters 
#during several trail attempts: namely 'Premiums in LOB: Motor','Premiums in LOB: Household','Premiums in LOB: Health',
#'Premiums in LOB:  Life','Premiums in LOB: Work Compensations')

# 3.2.1 - Select the features
featuresCons = data[['Premiums in LOB: Motor','Premiums in LOB: Household','Premiums in LOB: Health',
                     'Premiums in LOB:  Life','Premiums in LOB: Work Compensations']]

plotting(data=featuresCons)


# Using PCA to show 5 variables in a 2D space:
def pca_plotting(data,labels=None):
    plt.figure(num=None, figsize=(7, 7), dpi=80, facecolor='w', edgecolor='k')
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['Principal Component 1', 'Principal Component 2'])
    
    plt.scatter(principalDf.iloc[:, 0], principalDf.iloc[:, 1],c=labels, s=5, cmap='viridis')
    plt.xlabel(principalDf.columns[0], fontsize=10)
    plt.ylabel(principalDf.columns[1], fontsize=10)
    plt.show()

pca_plotting(featuresCons)    


# 3.2.2 - Scaling
#K-means clustering is "isotropic" in all directions of space and therefore tends to produce more or less round (rather than elongated) clusters.
#In this situation leaving variances unequal is equivalent to putting more weight on variables with smaller variance. So we scale the variables in order to remove the difference in magnitude of the clustering analysis.

scaled_featuresCons = StandardScaler().fit_transform(featuresCons.values)
scaled_featuresCons = pd.DataFrame(scaled_featuresCons, index=featuresCons.index, columns=featuresCons.columns)
scaled_featuresCons.describe()

# 3.2.3 - Elbow Graph
elbow_graph(scaled_featuresCons)
#The Elbow Graph seems to be indicating a number between 2 and 5 for number of clusters. Lets see what the Dendogram shows.

# 3.2.4 - Hierarchical (Dendogram)
#This might help us decide the number of clusters.

dendrogram_(scaled_featuresCons)

#Again, 3 seems to be the best number of clusters.

# 3.2.5 - K-Means

# Setting up the K-Means model:

kmeans_Cons = KMeans(n_clusters=2, max_iter=1000, random_state=12345, init = 'k-means++') # Set the number of clusters we want, the maximum number of iteration and a seed for the random state.
kmeans_Cons.fit(scaled_featuresCons)
labels_Cons = kmeans_Cons.labels_

# Plotting the resulting clusters
plotting(scaled_featuresCons,centers_cv=kmeans_Cons.cluster_centers_,labels=labels_Cons,x=0,y=3)

# Using PCA to show 5 variables in a 2D space:
pca_plotting(scaled_featuresCons,labels=labels_Cons)    






# 3.3 Clustering Categorical variables
# 3.3.1 - Select the features
featureModes = data[['Geographic Living Area', 'Has Children (Y=1)', 'Educational Degree']] # All categoric variables.
#Since these are categorical variables, there is no need of scaling.

#### 3.3.2 - K-Modes

#since in the other two cluster groups we chose 3 clusters, we should stick to this number of clusters now.
# Setting up K-Modes:
km = KModes(n_clusters=3, init='Huang', n_init=5, verbose=1)

labels_Cat = km.fit_predict(featureModes)
Counter(labels_Cat)







# 4 - Mean-Shift Clustering 
# 4.1 - Clusters "Customers value"
my_bandwidth = estimate_bandwidth(scaled_features_df,
                               quantile=0.2,
                               n_samples=1000)

ms = MeanShift(bandwidth=my_bandwidth,
               bin_seeding=True)

ms.fit(scaled_features_df)
labels_ms = ms.labels_
ms_cluster_centers = ms.cluster_centers_

# Plotting the resulting clusters

plt.figure(num=None, figsize=(7, 7), dpi=80, facecolor='w', edgecolor='k')

x = 0 # Select the variable that goes into the X axis in the graph.
y = 1 # Select the variable that goes into the Y axis in the graph.

plt.scatter(scaled_features_df.iloc[:, x], scaled_features_df.iloc[:, y], c=labels_ms, s=5, cmap='viridis')

centers_ms_cv = ms_cluster_centers
plt.scatter(centers_ms_cv[:, x], centers_ms_cv[:, y], c='black', s=200, alpha=0.5);
plt.xlabel(scaled_features_df.columns[x], fontsize=10)
plt.ylabel(scaled_features_df.columns[y], fontsize=10)
plt.show()

data.groupby(labels_ms).mean().loc[:,["Claims Rate", "Gross Monthly Salary", "TotalSpent", "Time"]]

#MS Clustering finds only two clusters when clustering customer value for the company.

#The found clusters are dividing the data points mainly based on Claims Rate, while just slightly considering Gross Monthly Salary into the clustering.
# 4.2 - "Consumption" Clusters
my_bandwidth = estimate_bandwidth(scaled_featuresCons,
                               quantile=0.2,
                               n_samples=1000)

ms = MeanShift(bandwidth=my_bandwidth,
               bin_seeding=True)

ms.fit(scaled_featuresCons)
labels_ms_Cons = ms.labels_
ms_cluster_centers = ms.cluster_centers_

# Plotting the resulting clusters

plt.figure(num=None, figsize=(7, 7), dpi=80, facecolor='w', edgecolor='k')

x = 0 # Select the variable that goes into the X axis in the graph.
y = 3 # Select the variable that goes into the Y axis in the graph.

plt.scatter(scaled_featuresCons.iloc[:, x], scaled_featuresCons.iloc[:, y], c=labels_ms_Cons, s=5, cmap='viridis')

centers_ms_con = ms_cluster_centers
plt.scatter(centers_ms_con[:, x], centers_ms_con[:, y], c='black', s=200, alpha=0.5);
plt.xlabel(scaled_featuresCons.columns[x], fontsize=10)
plt.ylabel(scaled_featuresCons.columns[y], fontsize=10)
plt.show()

#pca
plt.figure(num=None, figsize=(7, 7), dpi=80, facecolor='w', edgecolor='k')

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(scaled_featuresCons)
principalDf = pd.DataFrame(data = principalComponents, columns = ['Principal Component 1', 'Principal Component 2'])


plt.scatter(principalDf.iloc[:, 0], principalDf.iloc[:, 1], c=labels_ms_Cons, s=5, cmap='viridis')
centers = ms_cluster_centers
plt.xlabel(principalDf.columns[0], fontsize=10)
plt.ylabel(principalDf.columns[1], fontsize=10)
plt.show()

data.groupby(labels_Cat).mean().loc[:,["Premiums in LOB: Motor","Premiums in LOB: Household",
                                        "Premiums in LOB: Health",'Premiums in LOB:  Life',
                                        "Premiums in LOB: Work Compensations", "TotalSpent",
                                        "Claims Rate", "Gross Monthly Salary", "Time"]]

#Even though MS Clustering found 4 clusters for the Consumption variables, the cluster 0 holds more than 90% of all data, while the cluster 3 holds only 1 row.
#The graphs are not promising.
#These clusters cant separate much.
#Every result indicates that MS Clustering doest not work well with Consumption variables.
#In the end, MS CLustering does not seem like a good alternative for our project.


# 5 - Conclusions 
# 5.1 - Customer_Value Clusters
pd.crosstab(labels, data["Has Children (Y=1)"], normalize='index', rownames=["Customer_Value"])
#The only categorical variable that seems to have any connection to these clusters is the Has Children variable, 
#in which the cluster 2 shows a more balanced distribution of presence of children, while the clusters 0 and 1 show great tendency towards having children.    

data.groupby(labels).mean().loc[:,["Claims Rate", "Gross Monthly Salary", "Premiums in LOB: Motor", "Premiums in LOB: Health",
                                                    "Premiums in LOB: Work Compensations", "TotalSpent", "Time"]]

#We can see that these 4 clusters can divide the Claims Rate and Gross Monthly Salary very well, 
#which was the point of this clusterization. It barely has any effect on Premium variables and TotalSpent, 
#which was created using these variables. Has no effect over the Time variable whatsoever.

# 5.2 - Consumption Clusters
pd.crosstab(labels_Cons, data["Educational Degree"], normalize='index', rownames=["Consumption"])
pd.crosstab(labels_Cons, data['Has Children (Y=1)'], normalize='index', rownames=["Consumption"])

#It seems that both Educational Degree and Has Children show a different behavior when comparing the Consumption Clusters. While cluster 2 hold the majority of clients with Basic education and no PhD client, the cluster 0 seems to be the more educated one.

#Regarding children, the cluster 0 represents the clients that predominantly have children, while both other clusters show a more balanced distribution, even if a little more on the having children side.
data.groupby(labels_Cons).mean().loc[:,["Premiums in LOB: Motor","Premiums in LOB: Household",
                                        "Premiums in LOB: Health",'Premiums in LOB:  Life',
                                        "Premiums in LOB: Work Compensations", "TotalSpent",
                                        "Claims Rate", "Gross Monthly Salary"]]
#These clusters seem to be able to divide well all premiums variables as well as the TotalSpent variable.
#As expected, does not seem to divide the Claims Rate variable so well

# 5.3 - Demographic Clusters
data.groupby(labels_Cat).mean().loc[:,["Premiums in LOB: Motor","Premiums in LOB: Household",
                                        "Premiums in LOB: Health",'Premiums in LOB:  Life',
                                        "Premiums in LOB: Work Compensations", "TotalSpent",
                                        "Claims Rate", "Gross Monthly Salary", "Time"]]

#The clusters based on categorical data do not seem to divide any other variable meaningfully.


# Naming The Customer_Value Clusters
lab = np.repeat("aaaaaa a aaaa aaa a aaaaa aaaaaaaaa", labels.size)

for i in range(0, labels.size):
    if(labels[i]==0):
        lab[i] = "High CR - Low Sal - Many Children"
    elif(labels[i]==1):
        lab[i] = "Low CR - High Sal - Equal Children"
    elif(labels[i]==2):
        lab[i] = "High CR - High Sal - Equal Children" 
    else:
        lab[i] = "Low CR - Low Sal - Many Children"
        
lab

data.groupby(lab).mean().loc[:,["Claims Rate", "Gross Monthly Salary", "Premiums in LOB: Motor", "Premiums in LOB: Health",
                                                    "Premiums in LOB: Work Compensations", "TotalSpent", "Time"]]

# Naming The Consumption Clusters

lab_con = np.repeat("aaaaaa a aaaa aaa a aaaaa aaaaaaaa", labels_Cons.size)

for i in range(0, labels_Cons.size):
    if(labels_Cons[i]==0):
        lab_con[i] = "Low Spent - Motor - Many Children - "
    elif(labels_Cons[i]==1):
        lab_con[i] = "" 
    else:
        lab_con[i] = ""
        
lab_con

data.groupby(lab_con).mean().loc[:,["Claims Rate", "Gross Monthly Salary", "Premiums in LOB: Motor", "Premiums in LOB: Health",
                                                    "Premiums in LOB: Work Compensations", "TotalSpent", "Time"]]

# SUMMARY
# Customer_Value

# Plotting the resulting clusters:

plt.figure(num=None, figsize=(7, 7), dpi=80, facecolor='w', edgecolor='k')

x = 0 # Select the variable that goes into the X axis in the graph.
y = 1 # Select the variable that goes into the Y axis in the graph.
plt.scatter(scaled_features_df.iloc[:, x], scaled_features_df.iloc[:, y], c=labels, s=5, cmap='viridis')

centers_cv = kmeans.cluster_centers_
plt.scatter(centers_cv[:, x], centers_cv[:, y], c='black', s=200, alpha=0.5);
plt.xlabel(scaled_features_df.columns[x], fontsize=10)
plt.ylabel(scaled_features_df.columns[y], fontsize=10)

plt.show()

data.groupby(lab).mean().loc[:,["Claims Rate", "Gross Monthly Salary", "Premiums in LOB: Motor", "Premiums in LOB: Health",
                                                    "Premiums in LOB: Work Compensations", "TotalSpent", "Time"]]

