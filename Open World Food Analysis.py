#!/usr/bin/env python
# coding: utf-8

# # Import Packages and Dataset

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=SettingWithCopyWarning)
pd.set_option("display.max_rows", 150)
pd.set_option('display.max_columns', 150)


# In[2]:


cd Desktop\Nutrition Datasets


# In[3]:


df = pd.read_csv("en.openfoodfacts.org.products.tsv",
                       delimiter='\t',
                       encoding='utf-8')
df.head()


# # Exploring and Data Cleaning

# In[4]:


df.shape


# In[5]:


df.isnull().sum(axis=0)


# Quick glimpse of the missing data shows that some features have significant amount of null values (~300,000) out of 356027. Some do not even have a single value (356027). Therefore, it is completely reasonable to drop those columns.

# In[6]:


df = df.dropna(axis=1,how="all")
df.shape


# Another useful information from the columns is the fact that all the nutrients have the (_100g) suffix. Let's filter the nutrients and further investigate null values. 

# In[7]:


for column in df.columns:
    if '_100g' in column:
        print(column + ": " + str(df[column].isnull().sum()))


# Looks like many of the nutrient features have more than 1/2 of the values missing. Filling in the missing data with either the mean or the median is completely illogical. Therefore, we must delete these columns with significant amount of null values.

# In[8]:


missing = df.isnull().sum()
droplist = []
for ind, num in enumerate(missing):
    if '_100g' in missing.index[ind]:
        if num > (1/2)*df.shape[0]:
            droplist.append(missing.index[ind])
        
df = df.drop(columns=droplist)
print(df.shape, '\n')

for column in df.columns:
    if '_100g' in column:
        print(column + ": " + str(df[column].isnull().sum()))

df


# We can further filter the nutrition dataset by excluding the nutrition score and the saturated fat (since the total fat includes saturated fat).

# In[9]:


nutrition_features = ["energy_100g","fat_100g","carbohydrates_100g","sugars_100g","fiber_100g","proteins_100g",
                        "salt_100g","sodium_100g"]

nutrition = df[nutrition_features]
nutrition = nutrition.dropna(axis=0).reset_index(drop=True)
print("Percentage truncated nutrition dataset: ", (round(((len(df)-len(nutrition))/len(df)*100), 2)),"%", "\n")
nutrition


# Not too bad, after removing all the null values, there still is an ample amount of observations to work with.

# # Feature Engineering 

# Since, some of the nutrition features have been discarded, let's make some adjustments for couple of the features that are included. Furthermore, we can check for any possible error in data entry. 
# 
# The carbohydrates listed above is the total carbohydrates. Sugar and fiber are both sources of carbohydrates, so let's add a feature that accounts for the remainder as "r_carbohydrates".
# 
# The 3 macronutrients that make up the total energy content are fat, protein, and carbohydrates. Let's add a variable that has the calculated energy content as "calculated_energy".

# In[10]:


nutrition["calculated_energy"] = round(
    (nutrition.fat_100g*37) + 
    (nutrition.carbohydrates_100g*17) + 
    (nutrition.proteins_100g*17),
    2
)
nutrition["r_carbohydrates"] = round(nutrition.carbohydrates_100g - nutrition.sugars_100g - nutrition.fiber_100g, 2)


# There is one caveat to calculated_energy. Fiber is a tricky nutrient when it comes to determining whether it factors into the total carbohydrates, and thus, the total energy. The reason for this is that poorly digested foods such as fibers do not release as much energy. Furthermore, there is the matter of insoluble vs. soluble fiber where the former is not digested at all, hence, does not contribute to any calories while the latter is first digested by our gut bacteria to make short-chain fatty acids that do somewhat provide calories (according to the FDA, 2 calories per gram).
# 
# Since, we only know about the total fiber content and not its counterparts, we will use the calculated_energy column for any analysis performed.
# 
# Errors to consider:
# 
# (1.) Any products with weight greater than 100g. 
# 
# (2.) Any features with negative values (cannot have less of something that doesn't exist in the first place).
# 
# (3.) Any products with sugar greater than total carbohydrates, since, sugar is one part of carbohydrate that contains simple and complex carbs.
# 
# (4.) Any products exceeding the max energy value of 3700 (100% fat).
# 
# (5.) Any products with sodium greater than salt, since, sodium is one part of salt that contains chloride and other trace minerals.
# 
# (6.) Any products that have a higher "energy_100g" value than the "calculated_energy" value, since, the latter counts all types of carbohydrates (fiber) as 17 kJ. 

# In[11]:


negative = 0
for x in nutrition.values.flatten():
    if x < 0:
        negative += 1
if negative > 1:
    print("There are {} negative values.".format(negative))
else: 
    print("There are no negative values.")

print()
moresugar = np.where(nutrition['sugars_100g'] > nutrition['carbohydrates_100g'])
for sugar in moresugar:
    print("There are {} products with more sugar than carbohydrates.".format(len(sugar)),"\n")

overmax = nutrition.loc[nutrition.calculated_energy > 3700]
print("There are {} products that exceed the max energy (3700).".format(len(overmax)),"\n")

moresodium = np.where(nutrition['sodium_100g'] > nutrition['salt_100g'])
for sodium in moresodium:
    print("There are {} products with more sodium than salt.".format(len(sodium)),"\n")

energy = np.where(nutrition['energy_100g'] > nutrition['calculated_energy'])
for e in energy:
    print("There are {} products with more energy than they really should.".format(len(e)))


# There is a fair amount of errors in the composition of the products in the dataset. All but the energy error should be dropped from the dataset because, for the energy error, we were able to simply recalculate the total energy by adding all the macronutrients with the one assumption that all types of carbohydrates have some contribution to energy. As for errors that exceeded the max, we do not know what is contributing to cause that error due to us dropping features that may have allowed us to recalculate (but these were mostly empty anyways) or other unknown variables.

# In[12]:


nutrition = nutrition[nutrition >= 0].dropna(axis=0) # No negative values
nutrition = nutrition.loc[nutrition.carbohydrates_100g >= nutrition.sugars_100g] # Carbs must be > sugar.
nutrition = nutrition.loc[nutrition.calculated_energy <= 3700] # Max energy
nutrition = round(nutrition, 2) # Rounding all values to 2 decimal places


# In[13]:


print(nutrition.shape)
nutrition.head(n=3)


# Let us see if there indeed is a decrease in energy with increase in fiber.

# LINEAR REGRESSION

# In[14]:


from sklearn.linear_model import LinearRegression

def fiber_reg():
    X = nutrition.loc[:,"fiber_100g"].values.reshape(-1,1)
    y = nutrition.loc[:,"calculated_energy"].values


    reg = LinearRegression()
    reg.fit(X,y)
    reg.predict(nutrition.loc[:,"fiber_100g"].values.reshape(-1,1))

    fig = sns.regplot(X, y, fit_reg = False)
    plt.xlabel("Fiber")
    plt.ylabel("Energy")
    plt.show()
fiber_reg()


# No clear relationship is shown between fiber and energy. Other variables such as fat and proteins that contribute to total energy are masking the effect of fiber. However, the plot does show glaringly obvious outliers; 100g of fiber looks very suspicious. Even 60g of fiber is calling for further inspection.

# In[15]:


nutrition[nutrition.loc[:,"fiber_100g"] >=60].head()


# Looks like there is another error source we've overlooked. Look at the first example. There are as much fiber as the total carbohydrate, but the energy is very high! Even if we assumed earlier that all fiber will count as calories, if 99% of the total carbohydrate is fiber, the energy content should be realistically much lower. Also, there are products that have 0 energy which is impossible. 

# In[16]:


badfiber = list(nutrition.query("fiber_100g == carbohydrates_100g and carbohydrates_100g != 0").index)
badenergy = list(nutrition.query("carbohydrates_100g == 0").index)
dropindex = badfiber + badenergy
nutrition = nutrition.drop(index=dropindex)


# # Clustering 

# Before any analysis, let's see which products appear the most using word cloud.

# In[17]:


from wordcloud import WordCloud, STOPWORDS

stopwords=set(STOPWORDS)
nutrition["product"] = df.loc[nutrition.index, "product_name"] 
                                  
def cloudword(df):
    item = df["product"].apply(lambda l: str(l).lower().strip().split(','))
    df_item = item.apply(pd.Series).stack().reset_index(drop=True)
    text = " ".join(w for w in df_item)

    wordcloud = WordCloud(
        background_color="white",
        max_words = 30,
        max_font_size=30,
        min_font_size = 10
                         ).generate(text)

    plt.figure(figsize = (10,10))
    plt.imshow(wordcloud, interpolation ='bilinear')
    plt.axis("off")
    plt.show()
cloudword(nutrition)


# Look's like there are a lot of unknown products (represented by 'nan')! Get rid of those.

# In[18]:


nullindex = nutrition[nutrition["product"].isna()].index
nutrition = nutrition.drop(index=nullindex)


# In[19]:


nutrition["product"].value_counts().head()


# At first glimpse, one may notice the "inconsistency" of the results from the wordcloud and the value counts of the products. The reason for this is that a product may or may not have a unique name. What this means is that, even though "Potato Chips" appears 260 times in the dataset, taking dark chocolate as an example, that word may appear more than "Potato Chips" in the whole dataset BUT included in a unique name. Take a look at the code below. The results shows that the word "dark chocolate" appears more than "potato chips".

# In[20]:


ice_cream = 0
dark_chocolate = 0
potato_chips = 0
for item in (nutrition["product"].astype(str)):
    if 'ice cream' in item.lower():
        ice_cream += 1
    elif 'dark chocolate' in item.lower():
        dark_chocolate += 1
    elif 'potato chips' in item.lower():
        potato_chips += 1

print("Value counts\n")
print("Ice cream:",ice_cream)
print("Dark chocolate:",dark_chocolate)
print("Potato chips:",potato_chips)


# In[21]:


nutrition.head()


# In[22]:


from scipy import stats

threshold = 3
z = np.abs(stats.zscore(nutrition.loc[:,"fat_100g":"r_carbohydrates"]))
loc = np.where(z>3)
nutrition[(z>3)]


# SPLITTING DATASET

# In[23]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

train, test = train_test_split(nutrition, test_size = 0.3, random_state=42)

X_train = train.loc[:,"fat_100g":"r_carbohydrates"].values # Dropped the original energy column.
X_test = test.loc[:,"fat_100g":"r_carbohydrates"].values
y_train = train.loc[:,"product"]
y_test = test.loc[:,"product"]

# minmax = MinMaxScaler()
# X_train = minmax.fit_transform(X_train)
# X_test = minmax.fit_transform(X_test)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# KMEANS - HARD ASSIGNMENT

# In[24]:


from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

grid_params = {
    'n_clusters':np.arange(1,16),
    'init':['k-means++'],
    'random_state':[42],
    'algorithm':['full']
}
gs = GridSearchCV(
    estimator=KMeans(),
    param_grid=grid_params,
    cv=5
)
gs_results = gs.fit(X_train)
print("Best estimator:\n ",gs_results.best_estimator_)


# In[25]:


sumsq = []
K = range(1,16)
for k in K:
    km = KMeans(k)
    km = km.fit(X_train)
    sumsq.append(km.inertia_)

plt.plot(K, sumsq, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# The Grid Search showed that the more cluster the better, however, notice how the Elbow graph starts to plateau after 6 clusters. 12-14 clusters certainly does not have much different in sum of squared distances. For sake of interpretation and model complexity, 6 clusters will suffice.

# In[26]:


kmeans = KMeans(
    algorithm='full', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=6, n_init=10, n_jobs=None, precompute_distances='auto',
    random_state=42, tol=0.0001, verbose=0
)
kmeans= kmeans.fit(X_train)

train_kmeans = train.copy()
test_kmeans = test.copy()

train_kmeans["clusters"] = kmeans.predict(X_train)
test_kmeans["clusters"] = kmeans.predict(X_test)

sns.countplot(train_kmeans.clusters.values)
plt.xlabel("Clusters")

num_clusters = 6

for x in range(num_clusters):
    cloudword(train_kmeans[train_kmeans["clusters"]==x])
    


# GAUSSIAN MIXTURE MODEL - SOFT ASSIGNMENT

# In[27]:


from sklearn.mixture import GaussianMixture

aic_train = np.zeros(10)
bic_train = np.zeros(10)

components = range(10,110,10) 

for ind, com in enumerate(components):
    
    gmm = GaussianMixture(
    n_components = com,
    covariance_type = "full",
    n_init = 1,
    random_state=42
)
    gmm.fit(X_train)
    aic_train[ind]=gmm.aic(X_train)
    bic_train[ind]=gmm.bic(X_train)
    
    
fig, ax = plt.subplots()
ax.plot(components, aic_train, label='aic')
ax.plot(components, bic_train, label='bic')
ax.set_xlabel("Number of Gaussians")
ax.set_ylabel("Log-Likelihood (BIC)")
plt.legend()
plt.show()


# In[28]:


gmm = GaussianMixture(
    n_components = 25,
    covariance_type = "full",
    n_init = 1,
    random_state=42
).fit(X_train)

X_train_gmm = X_train.copy()
X_test_gmm = X_test.copy()

train_gmm = train.copy()
test_gmm = test.copy()

train_gmm["clusters"] = gmm.predict(X_train_gmm)
test_gmm["clusters"] = gmm.predict(X_test_gmm)

print("Converged: ",gmm.converged_,"\n")
print("Number of steps used to reach convergence: ",gmm.n_iter_,"\n")


# In[29]:


posteriorprobability = np.round(gmm.predict_proba(X_train_gmm), 2)
clusters_gmm = train_gmm.clusters.values
certainty = np.zeros(clusters_gmm.shape[0])

for x in range(len(clusters_gmm)):
    certainty[x] = posteriorprobability[x, clusters_gmm[x]]
train_gmm["certainty"] = certainty

# Percentage of certainty.
percentages = [.50,.75,.90]
for percent in percentages:
    print("Below {}%:".format(percent*100),round(len(train_gmm[train_gmm["certainty"] <= percent])/len(train_gmm),2))
    print("Above {}%:".format(percent*100),round(len(train_gmm[train_gmm["certainty"] > percent])/len(train_gmm),2))
    print()


# Our model is at least 50% certain for 98% of the all the datasets each observation assigned to each cluster! Even at 90% or above, at least 75% of all observations are at least 90% certain! 

# In[30]:


sns.countplot(train_gmm.clusters.values)
plt.show()


# In[31]:


clusterrange = range(0,25)
for x in clusterrange:
    print("Cluster {}".format(x))
    print(train_gmm["product"][train_gmm["clusters"] == x].value_counts().head())
    print()


# Turns out ice cream is a common product that is present in multiple clusters. One would ask the question, "Why not all ice cream in 1 cluster?" The reason is that these "ice cream" product may have different nutritional composition or even be just a part of a particular dish that includes ice cream. 
# 
# 
# The kmeans word clouds had the same result as well; 5 of the 6 clusters showed the prominence of ice cream for those clusters. Remember, the word clouds are created based on how many times a particular word is present in each of the cluster, not by the nutritional features. 
# 
# 
# So these clusters are clusterized based on the nutritional values. 
# 
# 
# Also, outliers or data entry errors by the user may be some of the possible reasons why the clusters with 'ice cream' are different.

# In[32]:


count = 0
for x,y in enumerate(train_gmm["product"]):
    if 'ice cream' in y.lower():
        print(train_gmm["product"].iloc[x],'\n')
        print(train_gmm.iloc[x],'\n\n')
        count += 1
        if count == 3:
            break


# If we compare these 3 different ice cream products, we can identify the difference in the nutritional values. All three are highly certain that they do in fact belong in the corresponding clusters with its own distribution (remember, gmm is fundamentally an algorithm for density estimation). Nevermind the micronutrients. Based on just the macronutrients, the difference is striking. Therefore, the clusters should be the same.

# In[33]:


icecream_clusters = [21,20,7]
for cluster in icecream_clusters:
    print("CLUSTER {}".format(cluster))
    print(train_gmm[train_gmm["clusters"]==cluster].describe(),'\n')


# I tried both the kmeans and gmm just for comparison purposes. Generally, kmeans is easy to apply and converge quickly, thus is also computationally faster than other EM (Expectation-Maximization) algorithms. However, since, kmeans is hard assignment and this may lead to overfitting especially when the cluster shape, size, and density are different, it is not the most optimal method of grouping observations especially when the clusters are not circular/spherical. Clustering via kmeans is done by minimizing the distance between the sample and the centroid, and at particular point, determine that an observation belongs to a particular cluster.
# 
# 
# On the other hand, gmm works well with different sizes and densities, since, it does not assume clusters to have any particular shape. For the dataset we are working with, we do not know which "type of food or department" each product belongs to, therefore, the data distribution is unknown (shape of the data is also unknown). Therefore, it would be more reasonable to calculate the probability that an observation belongs to a cluster and, subsequently, assign to a cluster after maximizing the likelihood. However, since we are dealing we probability, there may or may not be some degree of uncertainty.

# # Anomalies

# As mentioned before, there are bound to be outliers due to user error or products that naturally vary in nutrition. The ice cream example above showed this.

# In[34]:


gmm_logprob = gmm.score_samples(X_train_gmm)
threshold = 0.1

sns.distplot(gmm_logprob, kde=False, bins=50, color="Red")
g1 = plt.axvline(np.quantile(gmm_logprob, 0.25), color="Green", label="Q_25")
g2 = plt.axvline(np.quantile(gmm_logprob, 0.5), color="Blue", label="Q_50 - Median")
g3 = plt.axvline(np.quantile(gmm_logprob, 0.75), color="Green", label="Q_75")
g4 = plt.axvline(np.quantile(gmm_logprob, threshold), color="Purple", label="Q_ %i" % (int(threshold*100)))
handles = [g1, g2, g3, g4]
plt.xlabel("log-probabilities of the data spots")
plt.xlim((-25,40))
plt.ylabel("frequency")
plt.legend(handles) 
plt.show()


# In[35]:


def outliers(log_prob, threshold):
    epsilon = np.quantile(log_prob, threshold)
    outliers = np.where(log_prob <= epsilon, 1, 0)
    return outliers 

train_gmm["anomaly"] = outliers(gmm_logprob, threshold)
train_gmm[train_gmm["anomaly"] == 1]


# In[36]:


anomalies = train_gmm.groupby("clusters").anomaly.value_counts()
anomalies_cl = anomalies[:,1]

sns.barplot(x=anomalies_cl.index, y = anomalies_cl.values)
plt.show()


# In[37]:


train_gmm[(train_gmm["clusters"]==2) & (train_gmm["anomaly"]==1)]


# These anomalies are evidence that user errors are common in this dataset. For the barbeque sauces in the dataframe above, the sugar values are 0's for both which is impossible. 
# 
# 
# There are just way too many false information on the nutritional information of the products in the Open World Food dataset. 
