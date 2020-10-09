import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from titanic_eda import titanic_edl

###read in the file
file_name="train.csv"
titanic=pd.read_csv(file_name,header=0,sep=",")

###split data
X=titanic.drop("Survived",axis=1)
y=titanic["Survived"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42) #80% of training, 20% test

###inspect data

X_train.head()
titanic_edl=pd.concat([X_train, y_train], axis=1)
titanic_edl.head()

### inspect survival

titanic_edl[["Survived","PassengerId"]].groupby("Survived").nunique()

#### create 3 colmns : one for total family members and one for titles (embeded in names)
###one for cabins
titanic_edl['Title'] = titanic_edl['Name'].map(lambda x: x.split(', ')[1].split('.')[0])
titanic_edl['Title'].unique()

titanic_edl["Family_size"]=titanic_edl["SibSp"] + titanic_edl["Parch"]

titanic_edl["Level_cabin"]=titanic_edl["Cabin"].str[0]
titanic_edl["Level_cabin"][titanic_edl.Level_cabin.isna()]="No Cabin"


### let"s inspect size and count of data--> missing values
##but let"s go back to our X_train and y_train data set
X_train=titanic_edl
X_train.drop("Survived",axis=1,inplace=True)

X_train.count() #--> age has missing values




##### create a column transformer which does the following:
# - filles the missing values for Age
# - bin the follwoing columns:
#       1. Family_size
#       2. SibSp
#       
# - one hot encodes the following columns:
#       1. Pclass
#       2. Family_size
#       3. SibSp
#       4. Cabin levels 
#       5. Sex
#       6. Title
# - scale the whole numerical data with MinMax:
#       1. Fare
#       2. Age


#####first we need to create pipelines
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.metrics import f1_score, accuracy_score,precision_score, recall_score

def calculate_metric(model,X_train,y_true):
    y_pred=model.predict(X_train)
    acc=accuracy_score(y_pred,y_true)
    f1=f1_score(y_pred,y_true)
    precision=precision_score(y_pred,y_true)
    recall= recall_score(y_pred,y_true)
    
    return pd.DataFrame([model,acc, f1,precision,recall],index=['model',"accuracy","f1-score",'precision','recall'])



####binning categorical data:
# - Family size
# - SibSp
# 

X_train["fam_size"]=pd.cut(X_train["Family_size"], bins=3, labels=["small_fam","medium_fam","large_fam"])
X_train["Sib_Sp_num"]=pd.cut(X_train["SibSp"], bins=2, labels=["less_4","over_4"])




###make pipeline for Age
Age_pipe=make_pipeline(
    SimpleImputer(strategy="median"),
    MinMaxScaler()
)

Embarked_pip=make_pipeline(
    SimpleImputer(strategy="most_frequent"),
   OneHotEncoder()
)

### Assumptions:
# - Titles--> are  correlated to the gender and class(not including)
# - for example capt: 1 one capitan --> he dies --> dosn"t make sense to include it

# -Cabins--> are intresting, because cabin A has a low survival rate, even though it"s mostly
#           attributed to 1class passangers

# -family size--> important (smaller family sizes have higher survival rate)

###### column transformer based on the above pipelines
my_fet=ColumnTransformer(
    [ 
   ("impute then scale", Age_pipe, ["Age"]),
   ("one_hot_encode_rest", OneHotEncoder() ,["fam_size","Sib_Sp_num","Sex","Pclass","Level_cabin"]),
  ("scale",MinMaxScaler(),["Fare"]),
  ("impute than one hot encode",Embarked_pip,["Embarked"]),

    ]
)

#### fit and transform train data

my_fet.fit(X_train)
X_train_trans=my_fet.transform(X_train)
labels=["Age", "small_fam","medium_fam","large_fam","less_4","over_4","female","male","1class","2class","3class",
'C', 'No Cabin', 'B', 'F', 'D', 'E', 'A', 'G', 'T','Fare','E_S', 'E_C', 'E_Q',]

X_trans=pd.DataFrame(X_train_trans, columns=labels)

sn.heatmap(abs(X_trans.corr()))

##### creating intercation terms
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(interaction_only=True,include_bias = False)
#create intercation between female and family size
inter_female_family=poly.fit_transform(X_trans[["female","small_fam"]])[:,2]
inter_age_female=poly.fit_transform(X_trans[["female","Age"]])[:,2]
inter_fare_family_small=poly.fit_transform(X_trans[["Fare","small_fam"]])[:,2]

### create labels for this interactions
X_trans["fem_family"]=inter_female_family
X_trans["female_age"]=inter_age_female
X_trans["Fare_family"]=inter_fare_family_small

### plot the heatmap
sn.heatmap(abs(X_trans.corr()))

### inspect the correlations
import numpy as np
correlation_matrix=X_trans.corr()
correlation_matrix

####based on the correlation matrix we need to eliminate "less_4","over_4"
sn.heatmap(abs(pd.concat([X_trans,y_train], axis=1).corr()[['Survived']]),cmap="RdBu_r",center=0.0, annot=True)

###based on the correlations I will eliminate some redundant variables
# =less_4, over_2--> correlated with faimly size
# - 2class--> it is redundant
# - cabin C, E, A,T are irrelevant from corr with survival
# -embarked C is not relevant because its not correlated with survival
## -male doesn"t need to be in the feature set (because in this case gender is binary)

feature_subset=['Age', 'small_fam',  'large_fam', 
'female',  '1class',  '3class',  'No Cabin', 'B',
'F', 'D',  'G', 'Fare', 'E_S',  'E_Q', 'fem_family',
'female_age', 'Fare_family']
sn.heatmap(abs(X_trans[feature_subset].corr()))

sn.heatmap(abs(pd.concat([X_trans[feature_subset],y_train], axis=1).corr()[['Survived']]),cmap="RdBu_r",center=0.0, annot=True)


#### Logistic Regression
from sklearn.linear_model import LogisticRegression

###train FULL MODEL ############################

log_full=LogisticRegression()
log_full.fit(X_trans,y_train)

###balance classes
log_balanced=LogisticRegression(class_weight="balanced")
log_balanced.fit(X_trans,y_train)

### modify regularization --> performs worse
log_reg=LogisticRegression(C=0.4,class_weight="balanced")
log_reg.fit(X_trans,y_train)

###########baseline Modeling###############################

#####calculate metric for full model#######

models=[log_full,log_balanced,log_reg]

output=pd.Series([])
for model in models:
    out=calculate_metric(model=model,y_true=y_train, X_train=X_trans)
    output=pd.concat([out,output],axis=1)

output.dropna(axis=1, inplace=True)
output.columns=["log regulized","log balanced","log full"]
output
####log balanced dose the best!

####cross-validation of Full MODEL ###################
from sklearn.model_selection import cross_val_score

f1_score_ = cross_val_score(log_balanced, X_trans, y_train, cv=5, scoring='f1')
print("cross-validation scores", f1_score_)
print("Average f1 score: "+ str(f1_score_.mean()))
print("Std f1 score: " + str(f1_score_.std()))

accuracy = cross_val_score(log_balanced, X_trans, y_train, cv=5, scoring='accuracy')
print("cross-validation scores", accuracy)
print("Average accuracy: "+ str(accuracy.mean()))
print("Std f1 accuracy: " + str(accuracy.std()))


#####bootstrap of FULL MODEL############
from sklearn.utils import resample
def my_bootstrap(X_trans,y_train, log_balanced):
    boots = []
    f1=[]
    for i in range(1000):
        Xb, yb = resample(X_trans, y_train)
        log_balanced.fit(Xb, yb)
        y_pred=log_balanced.predict(Xb)
        score = log_balanced.score(Xb, yb)
        f1_out=f1_score(yb,y_pred)
        boots.append(score)
        f1.append(f1_out)
    return f1, boots


f1,boots=my_bootstrap(X_trans, y_train,log_balanced)

pd.Series(boots).hist()
plt.title("Accuracy distribution of log model, with all features")

# get percentiles for 90% confidence
f1.sort()
boots.sort()
ci80 = boots[100:-100]
print(f"80% confidence interval: {ci80[0]:5.2} -{ci80[-1]:5.2}")
f180=f1[100:-100]
print(f"80% confidence interval: {f180[0]:5.2} -{f180[-1]:5.2}")
ci90 = boots[50:-50]
f190=f1[50:-50]
print(f"90% confidence interval: {ci90[0]:5.2} -{ci90[-1]:5.2}")
ci95 = boots[25:-25]
print(f"95% confidence interval: {ci95[0]:5.2} -{ci95[-1]:5.2}")
f195 = f1[25:-25]

### output tables with baseline accuracy and f1 scores
indicatores=pd.DataFrame([["baseline",(ci80[0],ci80[-1]),(ci90[0],ci90[-1]),(ci95[0],ci95[-1])],
["baseline",(f180[0],f180[-1]),(f190[0],f190[-1]),(f195[0],f195[-1])]],index=[["accuracy","f1"]],columns=["model","80% interval","90% interval","95% interval"])


#### plotting baseline accuracy and f1 score
pd.Series(f1).hist()
plt.title("F1 score distribution of full model")
pd.Series(boots).hist()

##############selected features modeling############


###selecting a subset of features

log_subset_2=LogisticRegression(class_weight="balanced")
log_subset_2.fit(X_trans[feature_subset],y_train)


f1_sub,boots_sub=my_bootstrap(X_trans[feature_subset],y_train, log_subset_2)

f1_sub.sort()
boots_sub.sort()
ci80 = boots_sub[100:-100]
f180=f1_sub[100:-100]
ci90 = boots_sub[50:-50]
f190=f1_sub[50:-50]
ci95 = boots_sub[25:-25]
f195 = f1_sub[25:-25]

indicatores2=pd.DataFrame([["feature sel1",(ci80[0],ci80[-1]),(ci90[0],ci90[-1]),(ci95[0],ci95[-1])],
["feature sel1",(f180[0],f180[-1]),(f190[0],f190[-1]),(f195[0],f195[-1])]],index=[["accuracy","f1"]],columns=["model","80% interval","90% interval","95% interval"])

conf_intervals=pd.concat([indicatores,indicatores2])

#### compare histogram of full model vs model selection
pd.Series(boots_sub).hist()
plt.title("Accuracy distribution of selected features")
pd.Series(boots).hist()
pd.Series(f1).hist()
pd.Series(f1_sub).hist()
plt.title("F1 distribution of selected features")


#### find out the p-values of the log regression

###matrix outputs NAN because features are correlated
import statsmodels.discrete.discrete_model as sm
y_train=pd.DataFrame(y_train)
y_train.reset_index(inplace=True)
y_train=y_train.squeeze()["Survived"]


logit = sm.Logit(y_train,X_trans[feature_subset].dropna())
f = logit.fit()
print(f.params)
print(f.summary())
####based on the p-values I should exclude definetly:
# - small_fam (1) --> highy correlated with large family (skeptical)
#  - 1class (1)   --> skeptical 
# - B cabin (0.9)
# - E_S (embarked at S) (0.7)
# - No cabin (0.2)
# fem-family (1)
# F- cabin (0.2)

###Fare_family, 1class
feature_2=['Age',   'large_fam', 
'female',   '3class', "1class",
'D',  'G', 'Fare',  'E_Q',
'female_age',"Fare_family" ]

###inspect the p values again:
logit = sm.Logit(y_train,X_trans[feature_2].dropna())
f2 = logit.fit()
print(f2.params)
print(f2.summary())

#####let"s see what the sklearn model has to say about the new features

log_sub=LogisticRegression(class_weight="balanced")
log_sub.fit(X_trans[feature_2],y_train)


f1_sub2,boots_sub2=my_bootstrap(X_trans[feature_2],y_train, log_subset_2)
f1_sub2.sort()
boots_sub2.sort()

ci80 = boots_sub2[100:-100]
f180=f1_sub2[100:-100]
ci90 = boots_sub2[50:-50]
f190=f1_sub2[50:-50]
ci95 = boots_sub2[5:-5]
f195 = f1_sub2[25:-25]

indicatores3=pd.DataFrame([["feature sel2",(ci80[0],ci80[-1]),(ci90[0],ci90[-1]),(ci95[0],ci95[-1])],
["feature sel2",(f180[0],f180[-1]),(f190[0],f190[-1]),(f195[0],f195[-1])]],index=[["accuracy","f1"]],columns=["model","80% interval","90% interval","95% interval"])

conf_intervals=pd.concat([conf_intervals,indicatores3])
calculate_metric(model=log_sub,y_true=y_train, X_train=X_trans[feature_2])

####basicaly traded bias for more variance ???

####calculating metric for a diffrent regularization C
c=[1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
output2=pd.Series([])
for C in c:
    log_sub=LogisticRegression(class_weight="balanced",C=C)
    log_sub.fit(X_trans[feature_2],y_train)
    out=calculate_metric(model=log_sub,y_true=y_train, X_train=X_trans[feature_2])
    output2=pd.concat([out,output2],axis=1)

output2.dropna(axis=1, inplace=True)
output2

max(output2.loc["f1-score"])
#0.75-->c=0.2

max(output2.loc["recall"])
## c=0.4

max(output2.loc["precision"])
## c=0.2

###-> let"s choose a regularization constant C=0.4

log_sub_reg=LogisticRegression(class_weight="balanced",C=0.4)
log_sub_reg.fit(X_trans[feature_2],y_train)

f1_sub3,boots_sub3=my_bootstrap(X_trans[feature_2],y_train, log_sub_reg)


ci80 = boots_sub3[100:-100]
f180=f1_sub3[100:-100]
ci90 = boots_sub3[50:-50]
f190=f1_sub3[50:-50]
ci95 = boots_sub3[25:-25]
f195 = f1_sub3[25:-25]

indicatores4=pd.DataFrame([["feature sel2 reg",(ci80[0],ci80[-1]),(ci90[0],ci90[-1]),(ci95[0],ci95[-1])],
["feature sel2 reg",(f180[0],f180[-1]),(f190[0],f190[-1]),(f195[0],f195[-1])]],index=[["accuracy","f1"]],columns=["model","80% interval","90% interval","95% interval"])

conf_intervals=pd.concat([conf_intervals,indicatores4])

###the regularization is oo much!!!
    


##### Random forest
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier


#### Fit a tree
tree=DecisionTreeClassifier(random_state=43,max_depth=3)
tree.fit(X_trans,y_train)

tree.score(X_trans,y_train)

feature_importance=pd.DataFrame(tree.feature_importances_, index=X_trans.columns)
feature_importance.plot.barh()

plot_tree(tree, feature_names=labels,filled=True) 

calculate_metric(model=tree,y_true=y_train, X_train=X_trans)


#0.7963483146067416 --> with depth 2
#0.8342 --> max depth 3

### fit a tree for the selected sub features

tree_sub=DecisionTreeClassifier(random_state=43,max_depth=3,class_weight="balanced")
tree_sub.fit(X_trans[feature_2],y_train)

tree_sub.score(X_trans[feature_2],y_train)

feature_importance=pd.DataFrame(tree_sub.feature_importances_, index=feature_2)

feature_importance.plot.barh()
plot_tree(tree_sub, feature_names=labels,filled=True) 

calculate_metric(model=tree_sub,y_true=y_train, X_train=X_trans[feature_2])

#### precision and recall score are more baanced with the sub feature tree
### we trade a 0.1% of accuracy for that?? 



#### Random Forest
forest= RandomForestClassifier(n_estimators=1000,max_depth=3, random_state=43)
forest.fit(X_trans,y_train)
forest.score(X_trans,y_train)

feature_importance=pd.DataFrame(forest.feature_importances_, index=X_trans.columns)

feature_importance.plot.barh()

out_score_full=[]
for est in range(100, 1000,100):
    forest= RandomForestClassifier(n_estimators=est,max_depth=3, random_state=43)
    forest.fit(X_trans,y_train)
    score=forest.score(X_trans,y_train)
    out_score_full.append(score)
###100 trees are optimum

##random forest on a subset of the data
forest_sub= RandomForestClassifier(n_estimators=1000,max_depth=3, random_state=43)
forest_sub.fit(X_trans[feature_2],y_train)
forest_sub.score(X_trans[feature_2],y_train)

feature_importance=pd.DataFrame(forest_sub.feature_importances_, index=feature_2)

feature_importance.plot.barh()

####otimizing number of trees
out_score=[]
for est in range(100, 1000,100):
    forest_sub= RandomForestClassifier(n_estimators=est,max_depth=3, random_state=43)
    forest_sub.fit(X_trans[feature_2],y_train)
    score=forest_sub.score(X_trans[feature_2],y_train)
    out_score.append(score)

###optimzing for 100 trees


#### gets betetr with increasing depth--> tradeoff for overfitting
out_depth=[]
for est in range(2,6):
    forest_sub= RandomForestClassifier(n_estimators=est,max_depth=est, random_state=43)
    forest_sub.fit(X_trans[feature_2],y_train)
    score=forest_sub.score(X_trans[feature_2],y_train)
    out_depth.append(score)


######testing data

X_test['Title'] = X_test['Name'].map(lambda x: x.split(', ')[1].split('.')[0])

X_test["Family_size"]=X_test["SibSp"] + X_test["Parch"]

X_test["Level_cabin"]=X_test["Cabin"].str[0]
X_test["Level_cabin"][X_test.Level_cabin.isna()]="No Cabin"

X_test["fam_size"]=pd.cut(X_test["Family_size"], bins=3, labels=["small_fam","medium_fam","large_fam"])
X_test["Sib_Sp_num"]=pd.cut(X_test["SibSp"], bins=2, labels=["less_4","over_4"])





#### test intercations
X_test_trans=my_fet.transform(X_test)
X_test_trans=pd.DataFrame(X_test_trans,columns=labels)


#create intercation between female and family size
inter_female_family=poly.fit_transform(X_test_trans[["female","small_fam"]])[:,2]
inter_age_female=poly.fit_transform(X_test_trans[["female","Age"]])[:,2]
inter_fare_family_small=poly.fit_transform(X_test_trans[["Fare","small_fam"]])[:,2]

X_test_trans["fem_family"]=inter_female_family
X_test_trans["female_age"]=inter_age_female
X_test_trans["Fare_family"]=inter_fare_family_small

#### log model predictions
#calculate_metric(model=log_balanced,y_true=y_test, X_train=X_test_trans)
calculate_metric(model=log_sub,y_true=y_test, X_train=X_test_trans[feature_2])

#### bootstrap model
f1_test,boots_test=my_bootstrap(X_test_trans[feature_2],y_test, log_sub)
f1_test.sort()
boots_test.sort()

ci80 = boots_test[100:-100]
f180=f1_test[100:-100]
ci90 = boots_test[50:-50]
f190=f1_test[50:-50]
ci95 = boots_test[25:-25]
f195 = f1_test[25:-25]

indicatores5=pd.DataFrame([["test results",(ci80[0],ci80[-1]),(ci90[0],ci90[-1]),(ci95[0],ci95[-1])],
["test results",(f180[0],f180[-1]),(f190[0],f190[-1]),(f195[0],f195[-1])]],index=[["accuracy","f1"]],columns=["model","80% interval","90% interval","95% interval"])

pd.Series(boots_test).hist()
plt.title("Accuracy Distribution on the test data")
pd.Series(f1_test).hist()
plt.title("F1 and Accuracy Distribution on the test data")
#conf_intervals=pd.concat([conf_intervals,indicatores5])


### test out also teh balanced model
f1_test2,boots_test2=my_bootstrap(X_test_trans[feature_2],y_test, log_sub)
f1_test2.sort()
boots_test2.sort()

ci80 = boots_test2[100:-100]
f180=f1_test2[100:-100]
ci90 = boots_test2[50:-50]
f190=f1_test2[50:-50]
ci95 = boots_test2[25:-25]
f195 = f1_test2[25:-25]

indicatores6=pd.DataFrame([["test results",(ci80[0],ci80[-1]),(ci90[0],ci90[-1]),(ci95[0],ci95[-1])],
["test results",(f180[0],f180[-1]),(f190[0],f190[-1]),(f195[0],f195[-1])]],index=[["accuracy","f1"]],columns=["model","80% interval","90% interval","95% interval"])


###### training the model on the whole data

#transform X data

X['Title'] = X['Name'].map(lambda x: x.split(', ')[1].split('.')[0])

X["Family_size"]=X["SibSp"] + X["Parch"]

X["Level_cabin"]=X["Cabin"].str[0]
X["Level_cabin"][X.Level_cabin.isna()]="No Cabin"

X["fam_size"]=pd.cut(X["Family_size"], bins=3, labels=["small_fam","medium_fam","large_fam"])
X["Sib_Sp_num"]=pd.cut(X["SibSp"], bins=2, labels=["less_4","over_4"])



#### test intercations
X_full=my_fet.transform(X)
X_full_trans=pd.DataFrame(X_full,columns=labels)


#create intercation between female and family size
inter_female_family=poly.fit_transform(X_full_trans[["female","small_fam"]])[:,2]
inter_age_female=poly.fit_transform(X_full_trans[["female","Age"]])[:,2]
inter_fare_family_small=poly.fit_transform(X_full_trans[["Fare","small_fam"]])[:,2]

X_full_trans["fem_family"]=inter_female_family
X_full_trans["female_age"]=inter_age_female
X_full_trans["Fare_family"]=inter_fare_family_small

###fitting model on whole data
log_balanced.fit(X_full_trans,y)

### testing model outcome
f1_full,boots_full=my_bootstrap(X_full_trans,y, log_balanced)
ci80 = boots_full[100:-100]
f180=f1_full[100:-100]
ci90 = boots_full[50:-50]
f190=f1_full[50:-50]
ci95 = boots_full[25:-25]
f195 = f1_full[25:-25]

indicatores7=pd.DataFrame([["full calibration",(ci80[0],ci80[-1]),(ci90[0],ci90[-1]),(ci95[0],ci95[-1])],
["full calibration",(f180[0],f180[-1]),(f190[0],f190[-1]),(f195[0],f195[-1])]],index=[["accuracy","f1"]],columns=["model","80% interval","90% interval","95% interval"])

conf_intervals=pd.concat([conf_intervals,indicatores6])

####### apply whole model of the test data
test_data=pd.read_csv("test.csv",header=0,sep=",")

####transform the data

test_data['Title'] = test_data['Name'].map(lambda x: x.split(', ')[1].split('.')[0])

test_data["Family_size"]=test_data["SibSp"] + test_data["Parch"]

test_data["Level_cabin"]=test_data["Cabin"].str[0]
test_data["Level_cabin"][test_data.Level_cabin.isna()]="No Cabin"

test_data["fam_size"]=pd.cut(test_data["Family_size"], bins=3, labels=["small_fam","medium_fam","large_fam"])
test_data["Sib_Sp_num"]=pd.cut(test_data["SibSp"], bins=2, labels=["less_4","over_4"])



#### test intercations
test_data_full=my_fet.transform(test_data)
test_data_trans=pd.DataFrame(test_data_full,columns=labels)
test_data_trans["Fare"][test_data_trans.Fare.isna()]=test_data_trans["Fare"].mean()


#create intercation between female and family size
inter_female_family=poly.fit_transform(test_data_trans[["female","small_fam"]])[:,2]
inter_age_female=poly.fit_transform(test_data_trans[["female","Age"]])[:,2]
inter_fare_family_small=poly.fit_transform(test_data_trans[["Fare","small_fam"]])[:,2]

test_data_trans["fem_family"]=inter_female_family
test_data_trans["female_age"]=inter_age_female
test_data_trans["Fare_family"]=inter_fare_family_small

###predict outcome

y_pred=log_balanced.predict(test_data_trans)
y_pred=pd.DataFrame(y_pred, columns=["Survived"])
pred=pd.concat([test_data["PassengerId"],y_pred], axis=1)

pred.to_csv(r'C:/Users/Daniela/Documents/Spiced/Titanic predictions w02/prediction_titanic_D.csv', index = False, header=True)