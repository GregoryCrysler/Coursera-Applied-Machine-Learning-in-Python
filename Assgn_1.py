
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()


##########################################################################################################################################################################

DataDescr = cancer.DESCR

# You should write your whole answer within the function provided. The autograder will call
# this function and compare the return value against the correct solution value
def answer_zero():
    # This function returns the number of features of the breast cancer dataset, which is an integer. 
    # The assignment question description will tell you the general format the autograder is expecting
    return len(cancer['feature_names'])

# You can examine what your function returns by calling it in the cell. If you have questions
# about the assignment formats, check out the discussion forums for any FAQs
answer_zero() 


#####################################################################################
#Question 1

df = pd.concat([pd.DataFrame(cancer['data']),pd.DataFrame(cancer['target'])],axis=1)
df.columns = list(cancer['feature_names'])+list(['target'])

def answer_one():
    df = pd.concat([pd.DataFrame(cancer['data']),pd.DataFrame(cancer['target'])],axis=1)
    df.columns = list(cancer['feature_names'])+list(['target'])
    return df


answer_one()
#####################################################################################
#Question 2

#What is the class distribution? (i.e. how many instances of malignant (encoded 0) and how many benign (encoded 1)?)
#This function should return a Series named target of length 2 with integer values and index = ['malignant', 'benign']

cancerdf = answer_one()
cancerdf.replace(to_replace=[0,1],value=['malignant', 'benign'],inplace=True)
answer = pd.Series(cancerdf.groupby(['target']).size())

def answer_two():
    cancerdf = answer_one()
    cancerdf.replace(to_replace=[0,1],value=['malignant', 'benign'],inplace=True)
    answer = pd.Series(cancerdf.groupby(['target']).size())    
    return answer

answer_two()

#####################################################################################
#Split the DataFrame into X (the data) and y (the labels).

#This function should return a tuple of length 2: (X, y), where

#X, a pandas DataFrame, has shape (569, 30)
#y, a pandas Series, has shape (569,).

#cancerdf.shape




def answer_three():
    cancerdf = answer_one()
    X = cancerdf.iloc[:,:-1]
    y = cancerdf.iloc[:,-1]
    # Your code here    
    return X, y

answer_three()

#####################################################################################

#Question 4
#Using train_test_split, split X and y into training and test sets (X_train, X_test, y_train, and y_test).

#Set the random number generator state to 0 using random_state=0 to make sure your results match the autograder!

#This function should return a tuple of length 4: (X_train, X_test, y_train, y_test), where

#X_train has shape (426, 30)
#X_test has shape (143, 30)
#y_train has shape (426,)
#y_test has shape (143,)


from sklearn.model_selection import train_test_split

def answer_four():
    X, y = answer_three()
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = 0.75,random_state=0)
    # Your code here    
    return tuple([X_train, X_test, y_train, y_test])


#####################################################################################
#Question 5
#Using KNeighborsClassifier, fit a k-nearest neighbors (knn) classifier with X_train, y_train and using one nearest neighbor (n_neighbors = 1).

#This function should return a sklearn.neighbors.classification.KNeighborsClassifier.

from sklearn.neighbors import KNeighborsClassifier
​
def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    mdl = KNeighborsClassifier(n_neighbors=1)
    mdl.fit(X_train,y_train)
    # Your code here    
    return mdl


#####################################################################################
#Question 6
#Using your knn classifier, predict the class label using the mean value for each feature.

#Hint: You can use cancerdf.mean()[:-1].values.reshape(1, -1) which gets the mean value for each feature, ignores the target column, 
###and reshapes the data from 1 dimension to 2 (necessary for the precict method of KNeighborsClassifier).

#This function should return a numpy array either array([ 0.]) or array([ 1.])

def answer_six():
    cancerdf = answer_one()
    cancerdf_mean = cancerdf.mean()[:-1].values.reshape(1, -1)
    #X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    pred = knn.predict(cancerdf_mean)
    
    # Your code here
    
    return pred

#####################################################################################
#Question 7
#Using your knn classifier, predict the class labels for the test set X_test.

#This function should return a numpy array with shape (143,) and values either 0.0 or 1.0.


def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    preds = knn.predict(X_test)
    # Your code here    
    return preds.astype('float64')


#####################################################################################
#Question 8
#Find the score (mean accuracy) of your knn classifier using X_test and y_test.

#This function should return a float between 0 and 1

from sklearn.metrics import accuracy_score
def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    preds = answer_seven()
    # Your code here
    
    return accuracy_score(y_test,preds)





#def accuracy_plot():
#    import matplotlib.pyplot as plt

#    %matplotlib notebook

#    X_train, X_test, y_train, y_test = answer_four()

#    # Find the training and testing accuracies by target value (i.e. malignant, benign)
#    mal_train_X = X_train[y_train==0]
#    mal_train_y = y_train[y_train==0]
#    ben_train_X = X_train[y_train==1]
#    ben_train_y = y_train[y_train==1]

#    mal_test_X = X_test[y_test==0]
#    mal_test_y = y_test[y_test==0]
#    ben_test_X = X_test[y_test==1]
#    ben_test_y = y_test[y_test==1]

#    knn = answer_five()

#    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
#              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


#    plt.figure()

#    # Plot the scores as a bar chart
#    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

#    # directly label the score onto the bars
#    for bar in bars:
#        height = bar.get_height()
#        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
#                     ha='center', color='w', fontsize=11)

#    # remove all the ticks (both axes), and tick labels on the Y axis
#    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

#    # remove the frame of the chart
#    for spine in plt.gca().spines.values():
#        spine.set_visible(False)

#    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
#    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)