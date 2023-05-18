#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#Created on Fri May  5 22:16:50 2023

#@author: salamharoun
"""
import random
random.seed(15399972)
seed = 15399972 #my n number (private info please dont share)
#set random number generator here
#welcome to my capstone project!!!! im trying not to get totally f

#first i guess lets do dimensionality reduction cuz thats what pascal said

#brb let me upload the data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import PCA #This will allow us to do the PCA efficiently
from PIL import Image #These packages will allow us to do some basic
import requests #imaging of matrices and web scraping

art = np.genfromtxt('theArt.csv', delimiter = ',' , skip_header = 1) #fixed delimiter issue here

data = np.genfromtxt('theData.csv', delimiter = ',', skip_header = 1)

#first print data to see what it looks like
#print(data.shape) 
#shape of data is 299 rows and 221 columns -- nice

#okay now we do some eda (exploratoruy data analysis
#first look at raw data before we do pca to see if there are any problems

#since many questions exist in this data set, let's split it up a bit
#columns 0 to 90 show perference rating of art piece
plt.imshow(data[0:300][0:91]) # Visualize the array as an image (heatmap)
plt.xlabel('perference')
plt.ylabel('art')
plt.colorbar() # Add color bar 
#plt.show() 

#dont really notice anything here expcet that there is some missing data, it looks like many users liked 40
#some users liked all the art they saw (like user in row 35 ish)
#some users just hated all the art like uses in the 75 ish rows and uses past row 175

#doing some for the energy rating
plt.imshow(data[0:300][92:182]) # Visualize the array as an image (heatmap)
plt.xlabel('energy')
plt.ylabel('art')
plt.colorbar() # Add color bar 
#plt.show() 
#energy seems to be a lot calmer -> a lot of blue -> just looks like more blue version of perference
#they might be colinear variables>>>

#doing some for the dark personality traits
plt.imshow(data[0:300][182:194]) # Visualize the array as an image (heatmap)
plt.xlabel('answer (1-5)')
plt.ylabel('questions')
plt.colorbar() # Add color bar 
#plt.show()
#wow! looks like some of these are correlated already!the the same color
#tends to follow straight throough the questions, which means the users give similar answers
#to each question

#action perference eda
plt.imshow(data[0:301][194:205]) # Visualize the array as an image (heatmap)
plt.xlabel('answer (1-5)')
plt.ylabel('questions')
plt.colorbar() # Add color bar 
plt.show()
#some of these answers also tend to be correlated

#now self image eda
plt.imshow(data[0:301][205:215]) # Visualize the array as an image (heatmap)
plt.xlabel('answer (1-5)')
plt.ylabel('questions')
plt.colorbar() # Add color bar 
plt.show()
#missing data here
#a lot of dark in one spot, making me think some questions correlated

#not doing eda for last few columns because they need
#the other columsn before to make any sense of it
#and if i put all columns on the graph is the only way
#okay well im putting all columns on the graph
plt.imshow(data[0:301]) # Visualize the array as an image (heatmap)
plt.xlabel('all inputs')
plt.ylabel('all colums')
plt.colorbar() # Add color bar 
plt.show()
# a lot of them seem correlated in general since some lines
#just go all the way through the col

#okay now
#%%correlation matrix pca
#compute corrleation between each column acrosee all of these columns
corrMatrix = np.corrcoef(data,rowvar=False) #false cuz we want to know if the variables (questions, age, etc) are similar
#and not the individual users

# Plot the data:
plt.imshow(corrMatrix) 
plt.xlabel('variables')
plt.ylabel('variables')
plt.colorbar()
plt.show()

#OBSERVATIONS
#1. the yellowy green means their correleated -> some variables in the first 30 columns correleated
#2. variables in colum 75 area also correleated these are the artpieces which are similar
#3. overall some correleations exist here
#%%PCA
# z-scoring issue:

# pca algorithim assumes data given is normally distributed and z scored so we need to do that first
# if e dont z-score, cannot interpret data

#got to deal with nans before we can do this
#i think im going to drop rows with nan values because there are only 22 out of 300, which is only 7% of the total data, which I think
#i can justify
#i cannot delete nans because that would ruin the matrix
#i cant replace nans with out values, such as the mean because that would create disasterous complications if im going to cluster or use
#classification algorithims
#im gonna drop the nans
datA = np.copy(data)
datA = datA[~np.isnan(datA).any(axis=1), :]
#print(datA.shape)
#confirm that shape is 275 rows now, and 221 columns == nice


zscoredData = stats.zscore(datA) #Subtract the mean from each point, divide by SD, yields data with 
#mean 0 and SD = 1, in other words, z-scores. 

pca = PCA().fit(zscoredData) #Actually create the PCA object
pca2 = PCA().fit(datA) #Not z-scored - oh no - don't do this.
# z-scored and not z-scored data

#eigen values in order of decreasing magnitude
eigVals = pca.explained_variance_
eigVals2 = pca2.explained_variance_

#eigen vectors
loadings = pca.components_ #Rows: Eigenvectors. Columns: Where they are pointing
loadings2 = pca2.components_

rotatedData = pca.fit_transform(zscoredData) #this has the number of factors which are the numbers of columns
#we used to have 
#in order of decreasing eigenvectors, which are our principal components

varExplained = eigVals/sum(eigVals)*100 #eigen values are variance explained
#sum = 0
#variance explained for each factor
#for ii in range(len(varExplained)):
    #print(varExplained[ii].round(3))
    #sum += varExplained[ii].round(3)
    
#print(sum)
#okay so all the values add up to 100 kind of

#okay slay -- i aplogize for my silliness i got hit in the face last night and i feel woozy

#%%scree plot

#load questions here

numQuestions = 221 #number of columns here
x = np.linspace(1,numQuestions,numQuestions)
plt.bar(x, eigVals, color='gray')
plt.plot([0,numQuestions],[1,1],color='orange') # Orange Kaiser criterion line for the fox
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()

#damn. gotta make a choice here.
# alittle more than 50 principal compnents are above the kaiser criterion
# by elbow method, only one makes it since that is the biggest drop

kaiserThreshold = 1
#print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigVals > kaiserThreshold))
#64 by kaiser

threshold = 20 #90% is a commonly used threshold
eigSum = np.cumsum(varExplained) #Cumulative sum of the explained variance 
print('Number of factors to account for at least 90% variance:', np.count_nonzero(eigSum < threshold) + 1)
#omg 103 way too much

#since elbow method is simply visual and cannot calculate it, think im gonna
#go with kaiser criterion -> 64 factors WOW!

#64 is a lot to manually interpret
#we'll see how it goes

whichPrincipalComponent = 4 # Select and look at one factor at a time, in Python indexing
plt.bar(x,loadings[whichPrincipalComponent,:]*-1) # note: eigVecs multiplied by -1 because the direction is arbitrary
#and Python reliably picks the wrong one. So we flip it.
plt.xlabel('Question')
plt.ylabel('Loading')
plt.xticks(range(1, 240, 10))
plt.show() # Show bar plot

#pull up question meaning in another tab
#%%LOADINGS
#this is gonna be long
#PC1 -> points in questions 
#okay questions 37-91 point in same direction
# means painting 37-91 very similar -> can achieve same result by asking only
# about one of the paintings


#PC2
#92 column to 181 all point in same direction
#these questions were about energy rating of paintings, which are all
#correleated, meaning they give the same energry off
#better question vibe of all paintaings

#PC3
#162 to 171
#these columns are about the energy of some paintings
#this means some paintings are similar

#PC4
#201 -221 -
#loading is very small tho
#correlated with self image questions -> use just one question
#
#%%
#okay so we are doing pca for the entire dataset because its not really the same thing
#in addition, the pca wants us to reduce the first 91 columns, which are the art pieces, which are
#our data in another set,
#since the part pieces are rows(observations) in the other data set, we will not do pca on entire
#set
#we will do pca later
#%%
#QUESTION1
#at least our data is clean now
#Is classical art more well liked than modern art? 
# this involves user ratings so we cannot reduce the samples to the mean
#the mean is normalized sum and sum presumes that units we are summing
#are equal but psycholigcal distance between ratings not same
#no consistency
#mean doesn't mean anything here
#i think we will do man whitney u test to see if classical or modern art is better
#equal unit size doesn't matter since it depends on rank
#underlyind distribution also doesn't matter

plt.hist(datA.T[2],bins=9) #Ratings from 0 to 4 in 0.5 steps = 9 bins        

#okay, so the two groups are classical and modern
#there are not two columns, but many
#how do we solves this issue??
#so how do we represent the modern art in one column
#should we just call one colum "modern art" and add all the inputs together
classic = 0
modern = 0
for i in art.T[5]:
    if i == 1:
        classic +=1
    elif i == 2:
        modern +=1
#print (modern, classic)
#perfect! they are equal so this will work

#okay time to use indexing

#0 to 90 of data dataset has ratings -> want to go trhough these
#0 to 90 of rows are the art

#okay lets visualize this
modern_arT = []#!D array
classic_arT = []


count = 0
for i in art[:,5]:#the row here is the art and is looping through row
    if i == 1:
        for j in datA[:,count]:
            classic_arT.append(j)#count is the column of art
         #okay it seems to be using entire row
    elif i == 2:
        for j in datA[:,count]:
            modern_arT.append(j)
            
        
    count+=1
#print(count)
#print(len(classic_arT))
modern_art = np.copy(np.array(modern_arT))
classic_art = np.copy(np.array(classic_arT))

c_median = np.median(classic_art)
print("classic art median;", c_median)
m_median = np.median(modern_art)
print("modern art median:", m_median)

 # So in general, non-parametric analogues of these tests are more suitable
 # Test for comparing medians of ordinal data (such as movie ratings)
 # from 2 groups:
u1,p1 = stats.mannwhitneyu(modern_art,classic_art) #1 vs. 2

print("u = ", u1, " p-value = ", p1)

#do wilcxon rank sum

#um u value pretty huge, i wonder if the degrees of freedom are misconstrued here? i dont know if they use
#degrees of freedom

#i guess im rolling with this
#u value of 38814531.5
# p value of 1.3928200744791768e-87 --pretty unlikely due to chance alone
# i mean the data is pretty large so i guess

#anyway since we established a difference, let's see which is liked more...
# i dont know how to pick the better one i just know they are veryyyy different
# i guess ill just use medians



#print(modern_median, classic_median)
#classic music has greater median 
#anyway since u is positive, that means the second group is larger, which is classic art! great

#SURPRISE
#although mann whiteny u and wilcoxon very similar, the wilcoxon is for paired samples so we should
#use that instead
# or maybe not since they are not paired?


#%%
#Question 2

#Is there a difference in the preference ratings for modern art vs. non-human (animals and
#computers) generated art? 

#this question is very similar to the last one, therefore im going to use the same method:
#mann whiteny u test (how i love u so)

nonhuman = 0
modern = 0
for i in art.T[5]:
    if i == 3:
        nonhuman +=1
    elif i == 2:
        modern +=1
        
#print(nonhuman, modern)
#not nice -> nonhuman art is 21 while modern is 35...hopefully this doesn't create an issue

modern_arT = []#!D array
nonhuman_arT = []


count = 0
for col in datA.T[0:91]:
    if art[count][5] == 3:
        for i in col:
            nonhuman_arT.append(i) #okay it seems to be using entire row
    elif art[count][5] == 2:
        for i in col:
            modern_arT.append(i)
            
        
    count+=1
print("group modern art size: ",len(modern_arT))

modern_art = np.copy(np.array(modern_arT))
nonhuman_art = np.copy(np.array(nonhuman_arT))

#i realllyyyyy hope this doesn't create an issue

u1,p1 = stats.mannwhitneyu(nonhuman_art, modern_art) #1 vs. 2

print("u = ", u1, "p value = ", p1)
# u score is 19013098.0
# p-value is 1.3045360828419867e-243


#so yeah i guess there is a difference...
#%%QUESTION 3
#Do women give higher art preference ratings than men? 

#okay well let's see
#let's seperate data into women and men
women_ratingS = []
men_ratingS = []

for row in datA:
    if row[216] == 1:
        for r in row[0:91]:
            men_ratingS.append(r)
            
    if row[216] == 2:
        for r in row[0:91]:
            women_ratingS.append(r)
women_ratings = np.array(women_ratingS)
men_ratings = np.array(men_ratingS)

#mann whiteny u just shows if the stats are different, not higher

print("size of women population:",len(women_ratings))
print("size of men population:",len(men_ratings))

u1,p1 = stats.mannwhitneyu(women_ratings, men_ratings)           
    


print("u is ", u1, "p value is ", p1)
#u is 402630336
# p is 0.638599
#so not not statistically signifcant and no
#%%QUESTION 4
#Is there a difference in the preference ratings of users with some art background (some art
#education) vs. none?

#split them again and do another mann whitney u test
educated = []
not_educated = []
count = -1
for row in datA:
    count+=1
    if row[218] == 0:
        count = 0
        for r in row[0:91]:
            not_educated.append(r)
            count+=1
            if count == 91:
                break
        
    if row[218] > 0:
        count = 0
        for r in row[0:91]:
            educated.append(r)
            count +=1
            if count == 91:
                break

#okay mann whitney
u1,p1 = stats.mannwhitneyu(educated, not_educated)           
    


print("the u score is",u1)
print("the p value is",p1)
print("The size of the art educated group is", len(educated))
print("The size of the art non-educated group is", len(not_educated))


#yes very different -> 64294073.5 is u test
#p value 1.0879023187309865e-07

#%%QUESTION 5
#Build a regression model to predict art preference ratings from energy ratings only. Make sure
#to use cross-validation methods to avoid overfitting and characterize how well your model
#predicts art preference ratings.
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#okay sets seperate ratings into perference and energy
perference_ratings = datA[:,0:91]

energy_ratings = datA[:,91:182]

#okay perfect
#we need to use cross validation so keep that in mind
#model = LinearRegression().fit(energy_ratings, perference_ratings)
#rsq = model.score(energy_ratings, perference_ratings) 

#b0, b1 = model.intercept_, model.coef_ #Reach into the model object and get betas and intercept out
#print(b0, b1)
#yeah i figured, there's an issue with the data

#its not just one column its multiple, if im right there is 90
#print(len(b1))
#okay 221...
#which means the number of columns total
#275 rows
#i specificed which rows tho..lm check
#print(perference_ratings.shape)
#91 rows...and 221 columns

#okay fixed it to 275 rows and 91 columns

#okay there are 91 columns, how do we make them one column since the speicifc art doesnt matter

 
#will now combine columns using for loop
combined_perference_col = np.concatenate((datA[:,0:91]))

#print(len(combined_perference_col))

#it was 25025, which is 275 times 91 -> perfect!
combined_energy_col = np.concatenate((datA[:,91:182])).reshape(len(np.concatenate((datA[:,91:182]))), 1)

model = LinearRegression().fit(combined_energy_col, combined_perference_col)
rsq = model.score(combined_energy_col, combined_perference_col) 

b0, b1 = model.intercept_, model.coef_

#print(b0, b1, rsq)

#okay percept is 4.256
#coef is -0.00666
#r squared is 2.465504777382499e-05 -> not good

#i think im good use each user's median energy rating to predict each user's median preference rating
datA_copy = np.copy(datA)
energy = (datA_copy[:,91:182])
preference = (datA_copy[:,0:91])

median_energy = []
median_preference = []
for row in energy:
    med = np.median(row)
    median_energy.append(med)

for row in preference:
    med = np.median(row)
    median_preference.append(med)

median_energy = np.array(median_energy)
median_preference = np.array(median_preference)

#i wonder if i can get each paintings median and each paintings energy median
mean_energy_p = []
mean_preference_p = []
for i in datA.T[91:182]:
    med = np.mean(i)
    mean_energy_p.append(med)
    
for i in datA.T[0:91]:
    med = np.mean(i)
    mean_preference_p.append(med)
    
mean_preference_p = np.array(mean_preference_p)
mean_energy_p = np.array(mean_energy_p)
    
median_energy = []
median_preference = []
for row in energy:
    med = np.mean(row)
    median_energy.append(med)

for row in preference:
    med = np.mean(row)
    median_preference.append(med)

median_energy = np.array(median_energy)
median_preference = np.array(median_preference)

#x = median_energy.reshape(len(median_energy),1)
#y = median_preference

#x = combined_energy_col
#y = combined_perference_col

#x = stats.zscore(mean_energy_p.reshape(len(mean_energy_p), 1))
#y = stats.zscore(mean_preference_p)

x = stats.zscore(median_energy).reshape(len(median_energy),1)
y = stats.zscore(median_preference)

plt.plot(x,y,'o',markersize=5,color='black')  
plt.xlabel('energy')
plt.ylabel('preference')
#idk why data isn't working, obvi there is correlation here

#might delete an outlier because its skewing results and probably explain why
#might zscore data first
'''
for i in range(len(x)):
    if x[i] <-1.75:
        if y[i] <-1.5:
            print(i)
'''
#index of painting is 45
#

#split data for cross validation
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=(seed), shuffle=True)

model = LinearRegression()
scores = cross_val_score(model, X_train, y_train, cv=3, scoring = 'r2')

# Print the cross-validation scores and the mean score
print("Cross-validation scores:", scores)
print("Mean score:", scores.mean())

# Evaluate the model's performance on the testing set
model.fit(X_train, y_train)
test_score = model.score(X_test, y_test)
print("Testing score:", test_score)
#okay we need to cross validate
#i think we are doing k-fold cross validation using k=3-5 cuz datset not huge
rsq = model.score(x,y)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

plt.plot(x,y,'o',markersize=5,color='black')  
plt.xlabel('energy')
plt.ylabel('preference')
plt.plot(x, model.coef_*x + model.intercept_)

folds = 3
kf = KFold(n_splits=folds, shuffle=True, random_state= seed)

mse_values = []

#redefine model
model1 = LinearRegression()
#using mean squared error here because its associated with linear regression

for train_index, test_index in kf.split(x, y ):
    
    # Split the data into train and test sets for this fold
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model on the training data for this fold
    model1.fit(X_train, y_train)
    
    # Predict the target variable for the test data for this fold
    y_pred = model1.predict(X_test)
    
    # Calculate the mean squared error for this fold
    mse = mean_squared_error(y_test, y_pred)
    
    # Add the mean squared error to the list for this fold
    mse_values.append(mse)

mse_mean = np.mean(mse_values)
mse_std = np.std(mse_values)

print('Mean MSE: {:.2f}'.format(mse_mean))
print('MSE Standard Deviation: {:.2f}'.format(mse_std))

#idk if this will work since regression depends on continous data and this data is not


#%%QUESTION 6
#Build a regression model to predict art preference ratings from energy ratings and
#demographic information. Make sure to use cross-validation methods to avoid overfitting and
#comment on how well your model predicts relative to the “energy ratings only” model. 

#this time, we will call the predictors X

#two slices here
#slice1 = datA[:,91:182] #should we compress this too? i dont think so
#we have average user rating for each, and this is each user rating for each

#dont want to overfit so maybe get median score

energy_means = np.copy(datA)
count = 0
for row in energy_means[:,91:182]:
    mean = np.mean(row)
    energy_means[count,0] = mean
    count+=1
    
slice2 = datA[:,215:221] #demographics
slice1 = stats.zscore(energy_means[:,0]).reshape(-1,1)
X =  np.hstack((slice1, slice2)) #stacking them horzionitally
#print(slice1.shape)

#print(X.shape) #(275,96) -> perfect



 #
#maybe we need to do something to reduce perferences 
#perferences are not an one column but many
#Adding all perferences into one column gives us an error
#to predictor needs to be the same size as the predicted
#each datapoint needs to match
#could take mean of each perference???
#mean not really descriptive tho
#could take median
#i guess that is what we should do


#y = datA[:,0:91]

#still presents an issue here
# still need 300 rows
#all right we are using one cloumn from here that has mean rating of 
#all pieces for each user

#making copy of arry first of course
meanData = np.copy(datA)

#ok time to find med
count = 0
for row in meanData[:,0:91]:
    #print(len(row))
    mean = np.mean(row)
    meanData[count,0] = mean
    count+=1

y = stats.zscore(meanData[:,0])
#print(y.shape) #275 rows




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=(seed), shuffle=True)

model = LinearRegression()
scores = cross_val_score(model, X_train, y_train, cv=4, scoring = 'r2')

# Print the cross-validation scores and the mean score
print("Cross-validation scores:", scores)
print("Mean score:", scores.mean())

# Evaluate the model's performance on the testing set
model.fit(X_train, y_train)
test_score = model.score(X_test, y_test)
print("Testing score:", test_score)
#okay we need to cross validate
#i think we are doing k-fold cross validation using k=3-5 cuz datset not huge
print("Coefficients:", model.coef_)
b1 = model.coef_
print("Intercept:", model.intercept_)
b0 = model.intercept_
'''
plt.plot(X,y,'o',markersize=5,color='black')  
plt.xlabel('energy')
plt.ylabel('preference')
plt.plot(X, b1*X +b0)
'''
rSqrFull = model.score(X,y)

yHat = b1[0]*X[:,0] + b1[1]*X[:,1] + b1[2]*X[:,2] + b1[3]*X[:,3] + b1[4]*X[:,4] + b1[5]*X[:,5] + b0 #Evaluating the model: First coefficient times IQ value + 2nd coefficient * hours worked and so on, plus the intercept (offset)
plt.plot(yHat,X[:,5],'o',markersize=4) 
plt.xlabel('Prediction from model') 
plt.ylabel('Actual preference')  
plt.title('R^2 = {:.3f}'.format(rSqrFull))

#print(len(b1))
#okay makes sense since we have 96 columns predicting this
#print(rsq)
#okay so explains 51 percent of variance -> pretty good for social sciences

#ugh now we are doing k-fold cross validation
#tbh i dont understand how to do this idk if it was in the code session for linear
#regression but we can learn

folds = 5
kf = KFold(n_splits=folds, shuffle=True, random_state= seed)

mse_values = []

#redefine model
model1 = LinearRegression()
#using mean squared error here because its associated with linear regression

for train_index, test_index in kf.split(X, y ):
    
    # Split the data into train and test sets for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model on the training data for this fold
    model1.fit(X_train, y_train)
    
    # Predict the target variable for the test data for this fold
    y_pred = model1.predict(X_test)
    
    # Calculate the mean squared error for this fold
    mse = mean_squared_error(y_test, y_pred)
    
    # Add the mean squared error to the list for this fold
    mse_values.append(mse)

mse_mean = np.mean(mse_values)
mse_std = np.std(mse_values)

print('Mean MSE: {:.2f}'.format(mse_mean))
print('MSE Standard Deviation: {:.2f}'.format(mse_std))
#%%QUESTION 7
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#Considering the 2D space of average preference ratings vs. average energy rating (that
#contains the 91 art pieces as elements), how many clusters can you – algorithmically - identify
#in this space? Make sure to comment on the identity of the clusters – do they correspond to
#particular types of art?

#look at notes for clustering

#okay 
#find average perference rating and average energy rating for each painting

average_preference = []
for col in datA.T[0:91]:
    mean = np.mean(col)
    average_preference.append(mean)
    
average_energy = []
for col in datA.T[91:182]:
    mean = np.mean(col)
    average_energy.append(mean)
    
average_preference = np.array(average_preference)
average_energy = np.array(average_energy)
'''
plt.plot(average_preference,average_energy,'o',markersize=1)
plt.show()

#not a lot of data points but cool
# Format data:
x = np.column_stack((average_preference,average_energy))

# Fit model to our data:
dbscanModel = DBSCAN().fit(x) # Default eps = 0.5, min_samples = 5

# Get our labels for each data point:
labels = dbscanModel.labels_
print(labels)
'''
#okay so dbscan assumes that pca was done first but since we need all of the columns, we
#are not going to do that
#okay so this seems to be not working...maybe im doing something wrong...should do
#k-means

#first z-score data, not gonna do pca cuz its only two predictors
#not using pandas cuz no labels on this data anyway
zscoredPreference = stats.zscore(average_preference)

zscoredEnergy = stats.zscore(average_energy)

#zscoring cuz they use different scales and k means is a distanced based algorithm
x = np.column_stack((zscoredPreference, zscoredEnergy))


cluster_range = range(2,10)

silhouette_scores = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(x)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(x, labels))

# Plot the results
plt.plot(cluster_range, silhouette_scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

#omg this is so much more readable and interpretale wow
#from this i get 4

#gonna do elbow method now
wcss = [] # Within-Cluster-Sum-of-Squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=seed)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#elbow also shows 4 clusters where it starts to level off

#gonna do k means for that now
numClusters = 4
kMeans = KMeans(n_clusters = numClusters).fit(x) 
cId = kMeans.labels_ 
cCoords = kMeans.cluster_centers_ 

# Plot the color-coded data:
for ii in range(numClusters):
    plotIndex = np.argwhere(cId == int(ii))
    plt.plot(x[plotIndex,0],x[plotIndex,1],'o',markersize=1)
    plt.plot(cCoords[int(ii-1),0],cCoords[int(ii-1),1],'o',markersize=5,color='black')  
    plt.xlabel('preference')
    plt.ylabel('energy')
plt.show()
#there are four clusters, identy of each cluster
#orange would be high energy, low preference
#probably modern art, or non human
#middle energy middle preference probably a mix
#somewhat higher preference but high energy probably some
#classic art
#soothing high preference low energy probably classic art of nature
#idk how much though to put in this maybe do some more anlysis

#k gonna do that now
modern_arT = []#!D array
classic_arT = []


count = 91
for i in art[:,5]:#the row here is the art and is looping through row
    if i == 1:
        for j in datA[:,count]:
            classic_arT.append(j)#count is the column of art
         #okay it seems to be using entire row
    elif i == 2:
        for j in datA[:,count]:
            modern_arT.append(j)
            
        
    count+=1
#print(count)
#print(len(classic_arT))
modern_art_energy = np.copy(np.array(modern_arT))
classic_art_energy = np.copy(np.array(classic_arT))


print(np.median(modern_art), np.median(modern_art_energy))
print(np.median(classic_art), np.median(classic_art_energy))




'''
numClusters = 9 # how many clusters are we looping over? (from 2 to 10)
sSum = np.empty([numClusters,1])*np.NaN # init container to store sums


# Compute kMeans for each k:
for ii in range(2, numClusters+2): # Loop through each cluster (from 2 to 10)
    kMeans = KMeans(n_clusters = int(ii)).fit(x) # compute kmeans using scikit
    cId = kMeans.labels_ # vector of cluster IDs that the row belongs to
    cCoords = kMeans.cluster_centers_ # coordinate location for center of each cluster
    s = silhouette_samples(x,cId) # compute the mean silhouette coefficient of all samples
    sSum[ii-2] = np.sum(s) # take the sum
    # Plot data:
    plt.subplot(3,3,ii-1) 
    plt.hist(s,bins=20) 
    plt.xlim(-0.2,1)
    plt.ylim(0,50)
    plt.xlabel('Silhouette score')
    plt.ylabel('Count')
    plt.title('Sum: {}'.format(int(sSum[ii-2]))) # sum rounded to nearest integer
    plt.tight_layout() # adjusts subplot 

# Plot the sum of the silhouette scores as a function of the number of clusters, to make it clearer what is going on
plt.plot(np.linspace(2,numClusters,9),sSum)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of silhouette scores')
plt.show()
'''


#%%QUESTION 8
#Considering only the first principal component of the self-image ratings as inputs to a
#regression model – how well can you predict art preference ratings from that factor alone?

#okay we need to to do pca for self-image ratings
#omg my favorite part i lvoe working with psychology data

#use zscored data obvi

self_image_data = datA[:,205:215]
#print(self_image_data.shape)

#zscore now
self_image_zscored = stats.zscore(self_image_data)
#print(self_image_zscored.shape)

#okay do pca
pca = PCA().fit(self_image_zscored)

eigVals = pca.explained_variance_


#eigen vectors
loadings = pca.components_ #Rows: Eigenvectors. Columns: Where they are pointing


rotatedData = pca.fit_transform(self_image_zscored) #this has the number of factors which are the numbers of columns
#we used to have 
#in order of decreasing eigenvectors, which are our principal components

varExplained = eigVals/np.sum(eigVals)*100

#variance explained for each factor
#for ii in range(len(varExplained)):
    #print(varExplained[ii].round(3))
    #sum += varExplained[ii].round(3)
 
numQuestions = 10 #number of columns here
x = np.linspace(1,numQuestions,numQuestions)
plt.bar(x, eigVals, color='gray')
plt.plot([0,numQuestions],[1,1],color='orange') # Orange Kaiser criterion line for the fox
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()

whichPrincipalComponent = 1 # Select and look at one factor at a time, in Python indexing
plt.bar(x,loadings[whichPrincipalComponent,:]*-1) # note: eigVecs multiplied by -1 because the direction is arbitrary
#and Python reliably picks the wrong one. So we flip it.
plt.xlabel('Question')
plt.ylabel('Loading')
plt.show() # Show bar plot

#PCA#1
#reverse polarity ->2,6,8

#2 At times I think I am no good at all
#6 I certainly feel useless at times
#8 I wish I could have more respect for myself


#polarity -> 3,4,7

#3 I feel that I have a number of good qualities
#4 I am able to do things as well as most other people
#7 I feel that I'm a person of worth, at least on an equal plane with others

#so principal component 1 could be called...
#my general self-worth about myself (1 bad and 5 excellent)

#okay do linear regression on it now
#good ideea
data0 = np.copy(datA)
data1 = data0[:,0:91]
data2 =stats.zscore(data1)
means = []
#ok time to find med
count = 0
for row in data2:
    #print(len(row))
    mean = np.mean(row)
    data2[count][0] = mean
    means.append(mean)
    count+=1

means = np.array(means)
y = data2[:,0]
#gonna use the mean or median preference rating for each painting
#y = medianData[:,0]
x = rotatedData[:,0].reshape(len(rotatedData),1)
#ugh using standradized data doesnt change at all

#b0 = 4.349090909090909
#b1 = [0.00090635]
#rsq = 4.542805779217396e-06
#why is bad

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=(seed), shuffle=True)

model = LinearRegression()
scores = cross_val_score(model, X_train, y_train, cv=4, scoring = 'r2')

# Print the cross-validation scores and the mean score
print("Cross-validation scores:", scores)
print("Mean score:", scores.mean())

# Evaluate the model's performance on the testing set
model.fit(X_train, y_train)
test_score = model.score(X_test, y_test)
print("Testing score:", test_score)
#okay we need to cross validate
#i think we are doing k-fold cross validation using k=3-5 cuz datset not huge
print("Coefficients:", model.coef_)
b1 = model.coef_
print("Intercept:", model.intercept_)
b0 = model.intercept_

plt.plot(x,y,'o',markersize=5,color='black')  
plt.xlabel('self worth')
plt.ylabel('preference')
plt.plot(x, model.coef_*x + model.intercept_)
#cross validat
folds = 5
kf = KFold(n_splits=folds, shuffle=True, random_state= seed)

mse_values = []

#redefine model
model1 = LinearRegression()
#using mean squared error here because its associated with linear regression


for train_index, test_index in kf.split(x, y ):
    
    # Split the data into train and test sets for this fold
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model on the training data for this fold
    model1.fit(X_train, y_train)
    
    # Predict the target variable for the test data for this fold
    y_pred = model1.predict(X_test)
    
    # Calculate the mean squared error for this fold
    mse = mean_squared_error(y_test, y_pred)
    
    # Add the mean squared error to the list for this fold
    mse_values.append(mse)

mse_mean = np.mean(mse_values)
mse_std = np.std(mse_values)

print('Mean MSE: {:.2f}'.format(mse_mean))
print('MSE Standard Deviation: {:.2f}'.format(mse_std))
 #pretty good tbh
 
 #%%QUESTION 9
#Consider the first 3 principal components of the “dark personality” traits – use these as inputs
#to a regression model to predict art preference ratings. Which of these components
#significantly predict art preference ratings? Comment on the likely identity of these factors
#(e.g. narcissism, manipulativeness, callousness, etc.). 

#do pca for dark personailty traits
dark_personality_data = datA[:,182:194]
darkPersonalityZscore = stats.zscore(dark_personality_data)

pca = PCA().fit(darkPersonalityZscore)

eigVals = pca.explained_variance_


#eigen vectors
loadings = pca.components_ #Rows: Eigenvectors. Columns: Where they are pointing


rotatedData = pca.fit_transform(darkPersonalityZscore) #this has the number of factors which are the numbers of columns
#we used to have 
#in order of decreasing eigenvectors, which are our principal components

varExplained = eigVals/np.sum(eigVals)*100

#variance explained for each factor
#for ii in range(len(varExplained)):
    #print(varExplained[ii].round(3))
    #sum += varExplained[ii].round(3)
 
numQuestions = 12 #number of columns here
x = np.linspace(1,numQuestions,numQuestions)
plt.bar(x, eigVals, color='gray')
plt.plot([0,numQuestions],[1,1],color='orange') # Orange Kaiser criterion line for the fox
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()

whichPrincipalComponent = 1 # Select and look at one factor at a time, in Python indexing
plt.bar(x,loadings[whichPrincipalComponent,:]*-1) # note: eigVecs multiplied by -1 because the direction is arbitrary
#and Python reliably picks the wrong one. So we flip it.
plt.xlabel('Question')
plt.ylabel('Loading')
plt.show() # Show bar plot

#okay do regression for first 3 principal components
#maybe we should scale the y values because the x is scaled and just use mean
#okay
data0 = np.copy(datA)
data1 = data0[:,0:91]
data2 =stats.zscore(data1)

print(data2.shape)

means = []
#ok time to find med
count = 0
for row in data2:
    #print(len(row))
    mean = np.mean(row)
    data2[count][0] = mean
    means.append(mean)
    count+=1

means = np.array(means)
y = data2[:,0]
#y = means.T


x = rotatedData[:,0:2]

model = LinearRegression().fit(x, y)
rsq = model.score(x, y) 

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=(seed), shuffle=True)

model = LinearRegression()
scores = cross_val_score(model, X_train, y_train, cv=4, scoring = 'r2')

# Print the cross-validation scores and the mean score
print("Cross-validation scores:", scores)
print("Mean score:", scores.mean())

# Evaluate the model's performance on the testing set
model.fit(X_train, y_train)
test_score = model.score(X_test, y_test)
print("Testing score:", test_score)
#okay we need to cross validate
#i think we are doing k-fold cross validation using k=3-5 cuz datset not huge
print("Coefficients:", model.coef_)
b1 = model.coef_
print("Intercept:", model.intercept_)
b0 = model.intercept_



    
#b0 = 4.349090909090909
#b1 = 3.08491339e-05
#b2 = -8.10413624e-02
#b3 = -1.03979468e-01
#rsq = 0.028909629506538304
#not a great a model
folds = 5
kf = KFold(n_splits=folds, shuffle=True, random_state= seed)

mse_values = []

#redefine model
model1 = LinearRegression()
#using mean squared error here because its associated with linear regression

for train_index, test_index in kf.split(x, y ):
    
    # Split the data into train and test sets for this fold
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model on the training data for this fold
    model1.fit(X_train, y_train)
    
    # Predict the target variable for the test data for this fold
    y_pred = model1.predict(X_test)
    
    # Calculate the mean squared error for this fold
    mse = mean_squared_error(y_test, y_pred)
    
    # Add the mean squared error to the list for this fold
    mse_values.append(mse)

mse_mean = np.mean(mse_values)
mse_std = np.std(mse_values)

print('Mean MSE: {:.2f}'.format(mse_mean))
print('MSE Standard Deviation: {:.2f}'.format(mse_std))
#eh
#PC1
#revers polariry
#4,5,6,7
#normal poalrity -> 9,10,11,3
# inversely related to callousness and correctly correlated to narcissim

#pc2
#invers with 1,4,12
#corrleated with 7,8,9
#callousness

#pc3
#invers with 5,10
#corrleated with 8,2,3
#manipulativness

#%%which one explains variance most
y = data2[:,0]
print("narcissm varaince")
x = rotatedData[:,0].reshape(-1,1)
model = LinearRegression().fit(x, y)
rsq = model.score(x, y) 
print (rsq)
print()
print("callousness varaince")
x = rotatedData[:,1].reshape(-1,1)
model = LinearRegression().fit(x, y)
rsq = model.score(x, y) 
print (rsq)
print()
print("manipulativness varaince")
x = rotatedData[:,2].reshape(-1,1)
model = LinearRegression().fit(x, y)
rsq = model.score(x, y) 
print (rsq)

#%%QUESTION 10
#Can you determine the political orientation of the users (to simplify things and avoid gross
#class imbalance issues, you can consider just 2 classes: “left” (progressive & liberal) vs. “nonleft” (everyone else)) from all the other information available, using any classification model
#of your choice? Make sure to comment on the classification quality of this model. 

#any classification we want -> unsupervised i think since we dont have lables

#what is the rest of the info
#i guess action preferences are the rest
#gonna do k means

#its gonna be very dimensional

#okay gonna use the combined median rating for each user for both preference and 
#energy as predictors
median_energy = stats.zscore(median_energy).reshape(-1,1)
median_preference = stats.zscore(median_preference).reshape(-1,1)
gender = datA[:,216].reshape(-1,1)

x0 = np.hstack((median_energy, median_preference, gender))
#print (datA[:,182:216].shape, datA[:,218:222].shape )
x1 = np.hstack((datA[:,182:216], datA[:,218:222]))

x = np.hstack((x0,x1))
print(x.shape)

''' 
slice2 = datA[:,215:221] #demographics
slice1 = stats.zscore(energy_means[:,0]).reshape(-1,1)
X =  np.hstack((slice1, slice2)) #stacking them horzionitally
'''


#too many dimensions gonna do pca first
pca = PCA().fit(x)

eigVals = pca.explained_variance_


#eigen vectors
loadings = pca.components_ #Rows: Eigenvectors. Columns: Where they are pointing


rotatedData = pca.fit_transform(x) #this has the number of factors which are the numbers of columns
#we used to have 
#in order of decreasing eigenvectors, which are our principal components

varExplained = eigVals/np.sum(eigVals)*100

#variance explained for each factor
#for ii in range(len(varExplained)):
    #print(varExplained[ii].round(3))
    #sum += varExplained[ii].round(3)
 
numQuestions = 40 #number of columns here
x = np.linspace(1,numQuestions,numQuestions)
plt.bar(x, eigVals, color='gray')
plt.plot([0,numQuestions],[1,1],color='orange') # Orange Kaiser criterion line for the fox
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.show()

whichPrincipalComponent = 5 # Select and look at one factor at a time, in Python indexing
plt.bar(x,loadings[whichPrincipalComponent,:]*-1) # note: eigVecs multiplied by -1 because the direction is arbitrary
#and Python reliably picks the wrong one. So we flip it.
plt.xlabel('Question')
plt.ylabel('Loading')
plt.plot([0,numQuestions],[0.2,0.2],color='orange')
plt.plot([0,numQuestions],[-0.2,-0.2],color='orange')

plt.show() # Show bar plot

#get 5 for pca -> elbow method -> kasier criterion is 16 --> too much

#PCA1
#points in all directions except 3,38
#loading of 0.2 and above:
#4,6,9,21,24
# 4 -> manipulate others
# 6 flatter to get way
# 9 uncocnered with morality
# 21 walks in forst
# 24 ski
#14 presitge and status
#18 video games

#so how do we get this to one question...sking forest video game manipulation

#PCA2
#4,7,8,9,
#manipulation, exploit to own end, no remorse, morality none
#inverse with 16,18,21,23
#board games,video games, walks in forest, hike
#the more u hike, the more morality

#isolated activites no morality
#enjoy time with people make you sympathize with them?

#PCA3
#6,19,20,22
#flattery, yoga, meditate, beach walks
#18,17 #most pressure on 18
#role playing, 18 is video games

#do you ike video games and mind activties?

#PCA4
#2,26
#median preference, 
#8,21,23,24 (biggest), 37
#no remorse,walks in forest, hike, ski, user age

#do you prefer indoor act exibit instead of intense outdoor activies like hiking

#PCA5
#invers with 19,21, iggest loadings
#yoga and meditation # make people stressed less

#1,12,13,14
#energy, other admiration, attention,prestige
#sounds energy consuming to worry about this

#do i want other people to like me?

#PCA done now logistic regresstion with transformed data
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Creating a dataset
X = rotatedData[:,0:4]

#create y data
political_o = []
for i in datA[:,217]:
    if i <= 2:
        political_o.append(0)
    else:
        political_o.append(1)
y = np.array(political_o)

# Creating an instance of Logistic Regression Classifier
clf = LogisticRegression()

# Performing 5-fold cross-validation
scores = cross_val_score(clf, X, y, cv=5)

# Printing the accuracy scores for each fold
print("Accuracy scores for each fold: ", scores)

# Printing the mean accuracy score and standard deviation
print("Mean accuracy score: ", np.mean(scores))
print("Standard deviation: ", np.std(scores))





# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating an instance of Logistic Regression Classifier
clf = LogisticRegression()

# Training the model on the training data
clf.fit(X_train, y_train)

# Evaluating the model on the test data
score = clf.score(X_test, y_test)

# Printing the accuracy score
print("Accuracy score: ", score)


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, roc_auc_score


y_pred_proba = clf.predict_proba(X)[:, 1]

# Calculating the AUC score
auc = roc_auc_score(y, y_pred_proba)

# Printing the AUC score
print("AUC score: ", auc)

#plotting roc curve

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Create a logistic regression model and fit it to the training data
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predict probabilities of the testing set
y_proba = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

# Plot the ROC curve
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
#%%Extra credit
