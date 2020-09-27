
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
def part1_scatter():
    import matplotlib.pyplot as plt
    %matplotlib notebook
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4);
#####################################################################################
#Question 1
#Write a function that fits a polynomial LinearRegression model on the training data X_train for degrees 1, 3, 6, and 9. 
#(Use PolynomialFeatures in sklearn.preprocessing to create the polynomial features and then fit a linear regression model) 
#For each model, find 100 predicted values over the interval x = 0 to 10 (e.g. np.linspace(0,10,100)) and store this in a numpy array. 
#The first row of this array should correspond to the output from the model trained on degree 1, the second row degree 3, the third row degree 6, and the fourth row degree 9.



def answer_one():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn import linear_model  as lm

    # Your code here
    
    
    mdl_1 = lm.LinearRegression()
    pf_1 = PolynomialFeatures(1)
    X_pf_1 = pf_1.fit_transform(X_train.reshape(-1, 1))
    mdl_1.fit(X_pf_1,y_train)

    mdl_3 = lm.LinearRegression()
    pf_3 = PolynomialFeatures(3)
    X_pf_3 = pf_3.fit_transform(X_train.reshape(-1, 1))
    mdl_3.fit(X_pf_3,y_train)

    mdl_6 = lm.LinearRegression()
    pf_6 = PolynomialFeatures(6)
    X_pf_6 = pf_6.fit_transform(X_train.reshape(-1, 1))
    mdl_6.fit(X_pf_6,y_train)


    mdl_9 = lm.LinearRegression()
    pf_9 = PolynomialFeatures(9)
    X_pf_9 = pf_9.fit_transform(X_train.reshape(-1, 1))
    mdl_9.fit(X_pf_9,y_train)


    X_new = np.linspace(0,10,100)
    X_new_pf_1 = pf_1.fit_transform(X_new.reshape(-1, 1))
    X_new_pf_3 = pf_3.fit_transform(X_new.reshape(-1, 1))
    X_new_pf_6 = pf_6.fit_transform(X_new.reshape(-1, 1))
    X_new_pf_9 = pf_9.fit_transform(X_new.reshape(-1, 1))

    m1 = mdl_1.predict(X_new_pf_1)
    m3 = mdl_3.predict(X_new_pf_3)
    m6 = mdl_6.predict(X_new_pf_6)
    m9 = mdl_9.predict(X_new_pf_9)

    return np.vstack((m1,m3,m6,m9))

#####################################################################################
#X_train, X_test, y_train, y_test

#Question 2
#Write a function that fits a polynomial LinearRegression model on the training data X_train for degrees 0 through 9. 
#For each model compute the  R2R2  (coefficient of determination) regression score on the training data as well as the the test data, and return both of these arrays in a tuple.
#This function should return one tuple of numpy arrays (r2_train, r2_test). Both arrays should have shape (10,)

def answer_two():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn import linear_model  as lm
    from sklearn.metrics.regression import r2_score
    r2_train = []
    r2_test = []

    deg = np.linspace(0,9,10)
    for exp in deg:
        pf = PolynomialFeatures(np.int(exp))
        X_pf = pf.fit_transform(X_train.reshape(-1, 1))
        mdl = lm.LinearRegression()
        mdl.fit(X_pf,y_train)
        train_score = r2_score(y_train,mdl.predict(X_pf))
        r2_train.append(train_score)
        X_test_pf = pf.fit_transform(X_test.reshape(-1, 1))
        test_score = r2_score(y_test,mdl.predict(X_test_pf))
        r2_test.append(test_score)

    train_output = np.array(r2_train)
    test_output = np.array(r2_test)
    return tuple([train_output,test_output])


#####################################################################################
##Re-write Answer 3 based on this


#####################################################################################
#X_train, X_test, y_train, y_test

#Question 3
#Based on the  R2R2  scores from question 2 (degree levels 0 through 9), what degree level corresponds to a model that is underfitting? 
##What degree level corresponds to a model that is overfitting? What choice of degree level would provide a model with good generalization performance on this dataset?
#Hint: Try plotting the  R2R2  scores from question 2 to visualize the relationship between degree level and  R2R2 . 
##Remember to comment out the import matplotlib line before submission.

#This function should return one tuple with the degree values in this order: (Underfitting, Overfitting, Good_Generalization). There might be multiple correct solutions, however, you only need to return one possible solution, for example, (1,2,3).
def answer_three():
    outputs = answer_two()
    outputs_df = pd.DataFrame({'training_score':outputs[0], 'test_score':outputs[1]})
    outputs_df['TrainTestSpread'] = outputs_df.training_score - outputs_df.test_score
    over = outputs_df.sort_values('TrainTestSpread',ascending =False).index[0]
    under = outputs_df.sort_values('training_score',ascending =True).index[0]
    good = outputs_df.sort_values('TrainTestSpread',ascending =True).index[0]
    return tuple([under,over,good])


answer_three()
#####################################################################################

#X_train, X_test, y_train, y_test
#Question 4
#Training models on high degree polynomial features can result in overly complex models that overfit, so we often use regularized versions of the model to constrain model complexity
#   , as we saw with Ridge and Lasso linear regression.

#For this question, train two models, both on polynomial features of degree 12. : 
##a non-regularized LinearRegression model (default parameters) 
##and a regularized Lasso Regression model (with parameters alpha=0.01, max_iter=10000) 
##Return the  R2R2  score for both the LinearRegression and Lasso model's test sets.

#This function should return one tuple (LinearRegression_R2_test_score, Lasso_R2_test_score)




def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score


    pf = PolynomialFeatures(12)
    X_pf = pf.fit_transform(X_train.reshape(-1, 1))
    lmdl = LinearRegression()
    lmdl.fit(X_pf,y_train)

    rmdl = Lasso(alpha=0.01, max_iter=10000)
    rmdl.fit(X_pf,y_train)

    X_test_pf = pf.fit_transform(X_test.reshape(-1, 1))
    return tuple([r2_score(y_test,lmdl.predict(X_test_pf)), r2_score(y_test,rmdl.predict(X_test_pf))])    

answer_four()

#####################################################################################
    #import matplotlib.pyplot as plt
    ##%matplotlib notebook
    #plt.figure()
    #plt.scatter(deg, train_output, label='training data')
    #plt.scatter(deg, test_output, label='test data')
    #plt.legend(loc=4);


##########################################################################################################################################################################
Data = 'C:/SourceData/PythonClass'


mush_df = pd.read_csv(Data+'/mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2

#####################################################################################

#Question 5
#Using X_train2 and y_train2 from the preceeding cell, train a DecisionTreeClassifier with default parameters and random_state=0. 
##What are the 5 most important features found by the decision tree?
##As a reminder, the feature names are available in the X_train2.columns property, 
##and the order of the features in X_train2.columns matches the order of the feature importance values in the classifier's feature_importances_ property.
##This function should return a list of length 5 containing the feature names in descending order of importance.
##Note: remember that you also need to set random_state in the DecisionTreeClassifier.



def answer_five():
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train2,y_train2)

fi = pd.DataFrame([clf.feature_importances_,X_train2.columns]).transpose()
fi.reset_index(inplace=True)
fi.columns = ['idx','Importance','Column_Name']


fi.sort_values('Importance', axis=0, inplace=True, ascending=False)
return list(fi.iloc[:6,2])


#####################################################################################



#Question 6
#For this question, we're going to use the validation_curve function in sklearn.model_selection to determine training and test scores 
##for a Support Vector Classifier (SVC) with varying parameter values. 
##Recall that the validation_curve function, in addition to taking an initialized unfitted classifier object, takes a dataset as input and does its own internal train-test splits to compute results.
##Because creating a validation curve requires fitting multiple models, for performance reasons this question will use just a subset of the original mushroom dataset: 
#
###please use the variables X_subset and y_subset as input to the validation curve function (instead of X_mush and y_mush) to reduce computation time.
#The initialized unfitted classifier object we'll be using is a Support Vector Classifier with radial basis kernel. So your first step is to 
###create an SVC object with default parameters (i.e. kernel='rbf', C=1) and random_state=0. Recall that the kernel width of the RBF kernel is controlled using the gamma parameter.

#With this classifier, and the dataset in X_subset, y_subset, 
##explore the effect of gamma on classifier accuracy by using the validation_curve function to 
##find the training and test scores for 6 values of gamma from 0.0001 to 10 (i.e. np.logspace(-4,1,6)). 
###Recall that you can specify what scoring metric you want validation_curve to use by setting the "scoring" parameter. 
###In this case, we want to use "accuracy" as the scoring metric.
#For each level of gamma, validation_curve will fit 3 models on different subsets of the data, returning two 6x3 (6 levels of gamma x 3 fits per level) arrays of the scores for the training and test sets.
##Find the mean score across the three models for each level of gamma for both arrays, 
##creating two arrays of length 6, and return a tuple with the two arrays.

#X_subset, y_subset



def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    clf = SVC(random_state=0)
    gamma = np.logspace(-4,1,6)
    search = validation_curve(clf,X_subset,y_subset,'gamma',gamma,scoring='accuracy')
    tuple([search[0].mean(axis=1),search[1].mean(axis=1)])

    # Your code here

    return tuple([search[0].mean(axis=1),search[1].mean(axis=1)])
#####################################################################################

#Question 7
#Based on the scores from question 6, 
    ##what gamma value corresponds to a model that is underfitting (and has the worst test set accuracy)? 
    ##What gamma value corresponds to a model that is overfitting (and has the worst test set accuracy)? 
    ###What choice of gamma would be the best choice for a model with good generalization performance on this dataset (high accuracy on both training and test set)?

#Hint: Try plotting the scores from question 6 to visualize the relationship between gamma and accuracy. Remember to comment out the import matplotlib line before submission.
#This function should return one tuple with the degree values in this order: (Underfitting, Overfitting, Good_Generalization) Please note there is only one correct solution.



def answer_seven():  
    mdl_data = answer_six()
    outputs_df = pd.DataFrame({'training_score':mdl_data[0], 'test_score':mdl_data[1]})
    outputs_df['TrainTestSpread'] = np.abs(outputs_df.training_score - outputs_df.test_score)
    gamma = np.logspace(-4,1,6)
    over = gamma[outputs_df.sort_values('TrainTestSpread',ascending =False).index[0]]
    under = gamma[outputs_df.sort_values('training_score',ascending =True).index[0]]
    good = gamma[outputs_df.sort_values('TrainTestSpread',ascending =True).index[0]]
    return tuple([under,over,good])



    
#####################################################################################

#



#####################################################################################
import matplotlib.pyplot as plt
#%matplotlib notebook
plt.figure()
plt.scatter(gamma,mdl_data[0],label='training data')
plt.scatter(gamma,mdl_data[1], label='test data')
plt.legend(loc=4);