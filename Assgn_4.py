
import numpy as np
import pandas as pd

Data = 'C:/SourceData/PythonClass'

fdata = pd.read_csv(Data+'/fraud_data.csv')
##########################################################################################################################################################################

def answer_one():           
    return fdata.Class.sum() / fdata.shape[0]


#####################################################################################
# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv(Data+'/fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#####################################################################################

#Question 2
#Using X_train, X_test, y_train, and y_test (as defined above), 
##train a dummy classifier that classifies everything as the majority class of the training data. What is the accuracy of this classifier? What is the recall?

##This function should a return a tuple with two floats, i.e. (accuracy score, recall score).



def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score
    clf = DummyClassifier()
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    rec = recall_score(y_test,clf.predict(X_test))    
    return tuple([acc,rec])
#####################################################################################

#Question 3
#Using X_train, X_test, y_train, y_test (as defined above), train a SVC classifer using the default parameters. What is the accuracy, recall, and precision of this classifier?
#This function should a return a tuple with three floats, i.e. (accuracy score, recall score, precision score).


def answer_three():
    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC
    clf = SVC()
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    rec = recall_score(y_test,clf.predict(X_test))
    prc = precision_score(y_test,clf.predict(X_test))
    return tuple([acc,rec,prc])
#####################################################################################




#y_scores = clf.decision_function(X_twovar_test)
#precision, recall, thresholds = precision_recall_curve(y_test, y_scores)


#Question 4
#Using the SVC classifier with parameters {'C': 1e9, 'gamma': 1e-07}, what is the confusion matrix when using a threshold of -220 on the decision function. Use X_test and y_test.
#This function should return a confusion matrix, a 2x2 numpy array with 4 integers.
def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    from sklearn.metrics import precision_recall_curve
    clf = SVC(C=1e9, gamma=1e-07,probability=True)
    clf.fit(X_train, y_train)
    #precision, recall, thresholds = precision_recall_curve(y_test, clf.decision_function())
    cnf = confusion_matrix(y_test,(clf.decision_function(X_test)[:] >= -220).astype(int))
    return cnf

#####################################################################################


#Question 5
#Train a logisitic regression classifier with default parameters using X_train and y_train.

#For the logisitic regression classifier, create a precision recall curve and a roc curve using y_test and the probability estimates for X_test (probability it is fraud).

#Looking at the precision recall curve, what is the recall when the precision is 0.75?

#Looking at the roc curve, what is the true positive rate when the false positive rate is 0.16?

#This function should return a tuple with two floats, i.e. (recall, true positive rate).


#precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
def answer_five():
from sklearn import linear_model as lm
from sklearn.metrics import precision_recall_curve, roc_curve

clf = lm.LogisticRegression()
clf.fit(X_train,y_train)
roc_f, roc_t, roc_t = roc_curve(y_test,clf.predict_proba(X_test)[:,1]) #roc_f, roc_t,thrsesh
prc_p, prc_r, prc_t = precision_recall_curve(y_test,clf.predict_proba(X_test)[:,1])

prc = np.vstack([prc_p,prc_r]).transpose()
roc = np.vstack([roc_f,roc_t]).transpose()

rec = prc[np.round(prc[:,0],2) == 0.75,1][0]
tpr = roc[np.round(roc[:,0],2) == 0.16,1][0]
return tuple([rec,tpr]) 



#import matplotlib.pyplot as plt
#plt.figure()
#plt.xlim([-0.01, 1.00])
#plt.ylim([-0.01, 1.01])
#plt.plot(roc_f, roc_t, lw=3)#, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_f))
#plt.show()
#####################################################################################
#X_train, X_test, y_train, y_test
#Question 6
#Perform a grid search over the parameters listed below for a Logisitic Regression classifier, using recall for scoring and the default 3-fold cross validation.

#
#
#From .cv_results_, create an array of the mean test scores of each parameter combination




def answer_six():    

    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression()
    params = {'penalty': ['l1', 'l2'], 'C':[0.01, 0.1, 1, 10, 100]}
    grd = GridSearchCV(clf,params, scoring='recall',cv=3)
    grd.fit(X_train, y_train)

    scores = pd.DataFrame(grd.cv_results_)[['param_C', 'param_penalty','mean_test_score']]
    output = np.array(scores.pivot(index='param_C',columns='param_penalty',values='mean_test_score'))
    # Your code here
    
    return output


#def GridSearch_Heatmap(scores):
#    %matplotlib notebook
#    import seaborn as sns
#    import matplotlib.pyplot as plt
#    plt.figure()
#    sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
#    plt.yticks(rotation=0);
#    plt.show()

#GridSearch_Heatmap(answer_six())




from sklearn.model_selection import GridSearchCV

params = {'gamma': [0.01, 0.1, 1, 10], 'C':[0.01, 0.1, 1, 10]}
grd = GridSearchCV(m,params, scoring='recall',cv=3)
grd.fit(X_train, y_train)

    scores = pd.DataFrame(grd.cv_results_)[['param_C', 'param_penalty','mean_test_score']]
    output = np.array(scores.pivot(index='param_C',columns='param_penalty',values='mean_test_score'))
    # Your code here
    
    return output