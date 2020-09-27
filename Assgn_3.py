

#The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. 
##This is where predictive modeling comes in. 
##For this assignment, your task is to predict whether a given blight ticket will be paid on time.
#All data for this assignment has been provided to us through the Detroit Open Data Portal(https://data.detroitmi.gov/)


#We provide you with two data files for use in training and validating your models: train.csv and test.csv. 
#Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. 
#The target variable is compliance, 
##True if the ticket was paid early, on time, or within one month of the hearing data, 
##False if the ticket was paid after the hearing date or not at all, 
##and Null if the violator was found not responsible. 
###Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.

#Note: All tickets where the violators were found not responsible are not considered during evaluation. 
#They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
######################################################################################################################################################################################################
#probability that the corresponding blight ticket will be paid on time.
#The evaluation metric for this assignment is the Area Under the ROC Curve (AUC).
#Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.

#For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using readonly/train.csv. 
#Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from readonly/test.csv will be paid, and the index being the ticket_id.



#Hints
##Make sure your code is working before submitting it to the autograder.
##Print out your result to see whether there is anything weird (e.g., all probabilities are the same).
##Generally the total runtime should be less than 10 mins. You should NOT use Neural Network related classifiers (e.g., MLPClassifier) in this question.
##Try to avoid global variables. If you have other functions besides blight_model, you should move those functions inside the scope of blight_model.
##Refer to the pinned threads in Week 4's discussion forum when there is something you could not figure it out.




#def UnifyPredictors(DfTrain,DfTest):    
#    cols = list(set(DfTrain.columns).difference(set(DfTest.columns)).union(set(DfTest.columns).difference(set(DfTrain.columns))))
#    for c in cols:
#        try:
#            DfTrain = DfTrain.drop(labels=c,axis=1)
#        except:
#            pass
#        try:
#            DfTest = DfTest.drop(labels=c,axis=1)
#        except:
#            pass
#    return DfTrain,DfTest    

tsdf = feature_importances
tblname ='mdl'+str(iteration)+'_coef'

def ExportToSQL(tsdf,tblname):   
    tsdf.to_sql(tblname.split('.')[1],con=engine,if_exists='replace',schema=tblname.split('.')[0], chunksize=1000)
######################################################################################################################################################################################################



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

Data = 'C:/SourceData/PythonClass'

train = pd.read_csv(Data+'/train.csv', encoding = 'ISO-8859-1', low_memory=False,  dtype={'zip_code': pd.np.str,  'non_us_str_code': pd.np.str,  'grafitti_status': pd.np.str})
test = pd.read_csv(Data+'/test.csv', encoding = 'ISO-8859-1', low_memory=False)
test['compliance'] = np.NaN

#,'disposition'
cols = ['agency_name','state'
        ,'fine_amount'	,'admin_fee','state_fee'	,'late_fee','discount_amount','clean_up_cost','judgment_amount'
        ,'compliance']

train_df = train.loc[(np.isnan(train.compliance)==False) & (train.country == 'USA') & (~train.agency_name.isin(['Health Department','Neighborhood City Halls']) ),cols]
train_df['Train'] = 1
test_df = test[cols]
test_df['Train'] = 0

data_df = pd.concat([train_df,test_df],axis=0)
mdl_df = pd.get_dummies(data_df,drop_first =True)
###################################################################################################


pred_cols = [col for col in list(mdl_df.columns) if col != 'Train' and col != 'compliance']
             #.isin(['Source','compliance'])]
X = mdl_df.loc[mdl_df.Train == 1,pred_cols]
Y = mdl_df.loc[mdl_df.Train == 1,['compliance']]
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0,train_size=0.8)


###################################################################################################

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

iteration = 4
clf = GradientBoostingClassifier()


parameters = {'n_estimators': [50], 'max_depth':[4,6,8], 'learning_rate':[0.01,0.025, 0.05]    #, 'gamma': [0.00005, 0.00001]
              , 'subsample': [0.4,0.75,0.9]
              , 'max_features': [0.4,0.75]    #log2
              , 'min_samples_leaf': [25,50,100]
              , 'min_impurity_decrease':[1e-7,1e-3]
              } #

grd = GridSearchCV(clf,parameters, scoring='roc_auc',cv=3,n_jobs=-1,verbose=5,pre_dispatch='4*n_jobs')  #,pre_dispatch='4*n_jobs'
grd.fit(X_train, y_train)

joblib.dump(grd.best_estimator_, Data+'/mdl'+str(iteration)+'.pkl')
pd.DataFrame(grd.cv_results_).to_csv(Data+'/mdl'+str(iteration)+'.csv')


###################################################################################################
grd.score(X_test,y_test)


feature_importances = pd.DataFrame(grd.best_estimator_.feature_importances_, index = X_train.columns, columns = ['importance']).sort_values('importance',ascending=False)
feature_importances = feature_importances.reset_index(drop=False)
feature_importances.rename(columns = {'index':'features'}, inplace = True)
ExportToSQL(feature_importances,'AuthForecasting_Outpatient_TriWest.mdl'+str(iteration)+'_coef')




######################################################################################################################################################################################################

probs = grd_hist.predict_proba(X)[:,1]
preds = pd.concat([test.ticket_id,pd.Series(probs)],axis=1)





preds = pd.Series(preds,index='ticket_id')


#pred_cols = [col for col in list(mdl_df.columns) if col != 'Train' and col != 'compliance']
             #.isin(['Source','compliance'])]
X = mdl_df.loc[mdl_df.Train == 0,pred_cols]



grd_hist = joblib.load(Data+'/mdl3.pkl')



grd_hist.predict_proba(X)
pd.DataFrame(grd_hist.feature_importances_, index = X_train.columns, columns = ['importance']).sort_values('importance',ascending=False)
preds = grd_hist.predict(X)


tmp = grd.predict(X)









########################################################################################################################################################################################################




import pyodbc
import sqlalchemy as sql
import pandas.io.sql as psql

cnxn1 = pyodbc.connect('DSN=FSC_Analytics_FBCSDev2')
cursor = cnxn1.cursor()

engine = sql.create_engine("mssql+pyodbc://gcrysler@FSC_Analytics_FBCSDev2")

test.to_sql('test',con=engine,if_exists='replace',schema='AuthForecasting_Outpatient_TriWest', chunksize=1000, index=False)   ##
train.to_sql('train',con=engine,if_exists='replace',schema='AuthForecasting_Outpatient_TriWest', chunksize=1000, index=False)   ##


cols = [col for col, dtype in dict(train.dtypes).items() if dtype != 'object']
df = train[cols]
df.fillna(value=-1,axis=0,inplace=True)

df = pd.read_csv(Data+'/train_quant.csv')
tsql_chunksize = 2097 // len(df.columns)
tsql_chunksize = 1000 if tsql_chunksize > 1000 else tsql_chunksize  # cap at 1000 (limit for number of rows inserted by table-value constructor)
df.to_sql('train',con=engine,if_exists='replace',schema='AuthForecasting_Outpatient_TriWest', chunksize=1000, index=False)   ##



cols = [col for col, dtype in dict(test.dtypes).items() if dtype != 'object']
df = test[cols]
df.fillna(value=-1,axis=0,inplace=True)
df.to_sql('test',con=engine,if_exists='replace',schema='AuthForecasting_Outpatient_TriWest', chunksize=tsql_chunksize, index=False)   ##
#test[cols].to_sql('test',con=engine,if_exists='replace',schema='AuthForecasting_Outpatient_TriWest', chunksize=tsql_chunksize, index=False)    #, chunksize=1000)



