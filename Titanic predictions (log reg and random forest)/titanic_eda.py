import pandas as pd
import seaborn as sn
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


##read in the file
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


###percentage of survived customers from 1st class
first_class=titanic_edl["Pclass"]==1
survived_1=titanic_edl[["Pclass","Survived","PassengerId"]].loc[first_class].groupby("Survived").nunique()["PassengerId"]
survived_1_total=titanic_edl[["Pclass","Survived","PassengerId"]].loc[first_class].nunique()["PassengerId"]
prop_survived_1=survived_1/survived_1_total*100
print(prop_survived_1)

#####bar plot showing proportion of males to females over all classes

titanic_edl[["Pclass","Sex","PassengerId"]].groupby(["Pclass","Sex"]).count().plot.bar()
male_female=titanic_edl[["Pclass","Sex","PassengerId"]].groupby(["Pclass","Sex"]).count()
male_female.reset_index(inplace=True)

###plot with seaborn males/females

plt.figure(figsize=(5,5))
sn.barplot(x="Pclass",y="PassengerId",hue="Sex",data=male_female)
plt.xlabel("Class")
plt.ylabel("Number of passengers")
plt.title("Gender distribution")
plt.show()

###females outnumber males in all classes, except the 3ed class, where males are more
#than double the number of females

####survival of females and males in all classes

grouped_=titanic_edl[["Pclass","Sex","PassengerId","Survived"]].groupby(["Pclass","Sex","Survived"]).count()
grouped_.reset_index(inplace=True)

###find distribution of total survived? total non survived over class and sex
grouped_["SV_NSV"]=grouped_[["Survived","PassengerId"]].groupby("Survived").transform("sum")

grouped_["prop_surv"]=grouped_["PassengerId"]/grouped_["SV_NSV"]*100

grouped_.head()

#### plot survival based on sex over all classes, sum of probability of survival is 1
plt.figure(figsize=(5,5))
sn.barplot(x="Pclass",y="prop_surv",hue="Sex",data=grouped_[grouped_["Survived"]==1])
plt.xlabel("Class")
plt.ylabel("Survival rate")
plt.title("Survival rate based on classe for all genders)")
plt.show()
###when it comes to survival rates: females again have a higher survival rate than men,
# even in the 3ed class (by 5%), where men outnumber women 
#women survival is actually almost more tahn double that of men



#### calculate how many % of women actually survived vs men 
grouped_["Total_gender"]=grouped_[["Sex","PassengerId"]].groupby("Sex").transform("sum")
grouped_["Surv_sex"]=grouped_["PassengerId"]/grouped_["Total_gender"]*100


###plot survival rates based on gender

plt.figure(figsize=(5,5))
sn.boxplot(x="Sex",y="Surv_sex",hue="Survived", data=grouped_)
plt.xlabel("Sex")
plt.ylabel("Survival rate")
plt.title("Survival distribution based on gender (over all classes)")
plt.show()

#### age and fare

titanic_edl[["Survived","Age"]].groupby(["Survived"]).mean()
### on average the lower the age the higher the survival chances were (28.4)

titanic_edl[["Survived","Age","Sex"]].groupby(["Survived","Sex"]).mean().unstack().plot.bar()
plt.title("Age distribution per gender and survival")
plt.show()
#### overall age mean for surviving women passangers was higher than that of surviving male passangers,
# but also higher than that of non surviving females (which is strange ).
# basically age for women is directly prop to survival rate . for men the distribution is
# as expected --> namly older men died whilst younger survived

titanic_edl[["Survived","Age","Pclass"]].groupby(["Survived","Pclass"]).mean().unstack().plot.bar()
plt.ylabel("Age")
plt.xlabel("Survived, Class")
plt.title("Survived per age and class")
plt.show()
### this looks a bit more "normal": the survival age increases by class and is usually lower than the age on non survival

####let"s look at the age distribution for each passenger class
#We can set dodge as True in the swarmplot to split the distributions
fig, ax = plt.subplots(figsize=(12,6), nrows=1, ncols=3)
plt.suptitle('Age Distribution of Survivor state by Gender in Each Class', fontsize=16)

for i in range(3):
    ax[i].set_title('Class {}'.format(i+1))
    ax[i].set_ylim(-5,85)
    sn.boxplot(data=titanic_edl[titanic_edl['Pclass']==i+1],
                  x='Survived',
                  y='Age',
                  hue='Sex',
                  hue_order=['male','female'],
                  dodge=True,
                  ax=ax[i])

ax[1].set_ylabel(None)
ax[2].set_ylabel(None)

ax[0].legend_.remove()
ax[1].legend_.remove()

##lets look at prices 
titanic_edl[["Survived","Fare"]].groupby(["Survived"]).mean()

titanic_edl[["Survived","Fare","Sex"]].groupby(["Survived","Sex"]).mean().unstack().plot.bar()
plt.ylabel("Fare Prices")
plt.title("Average Fare price per gender and survival")
plt.show()

### fare prices for females who survived, where higher than those of men who survived


### fare price seems to be correlated to more than just class,
# # since females are less in absolute numbers, but whit higher fare rates
# # men in fisrt class are more than double than women in fisrt class 

titanic_edl[["Survived","Fare","Sex","Pclass"]].groupby(["Survived","Sex","Pclass"]).mean().unstack().plot.bar(legend=False)
plt.ylabel("Fare")
plt.title("Fare Price distributed across survival state, per gender and class")
## men the ones that survive consitantly outpay the ones that don"t

##Let"s look at the distribution of Fare prices accross classes and genders
fig, ax = plt.subplots(figsize=(12,6), nrows=1, ncols=3)
plt.suptitle('Fare Price Distribution of Survivor state by Gender in Each Class', fontsize=16)

for i in range(3):
    ax[i].set_title('Class {}'.format(i+1))
    ax[i].set_ylim(-5,260)
    sn.boxplot(data=titanic_edl[titanic_edl['Pclass']==i+1],
                  x='Survived',
                  y='Fare',
                  hue='Sex',
                  hue_order=['male','female'],
                  dodge=True,
                  ax=ax[i])

ax[1].set_ylabel(None)
ax[2].set_ylabel(None)

ax[0].legend_.remove()
ax[1].legend_.remove()

##lets look at prices 
titanic_edl[["Survived","Fare"]].groupby(["Survived"]).mean()

titanic_edl[["Survived","Fare","Sex"]].groupby(["Survived","Sex"]).mean().unstack().plot.bar()
plt.ylabel("Fare Prices")
plt.title("Average Fare price per gender and survival")
plt.show()


### let"s see the connection amongst sex and sib/spouses and fare and sib/spuses

titanic_edl[["Survived","SibSp","PassengerId"]].groupby(["Survived","SibSp"]).count().unstack().plot.bar()
###survival with up to 4 siblings/spuses (small families)
#the most who survived where alone
###survival with up to 4 siblings/spuses (small families)




titanic_edl[["Survived","SibSp","Sex"]].groupby(["Survived","Sex"]).count()

titanic_edl[["Survived","SibSp","Sex","Fare"]].groupby(["Survived","SibSp","Sex"]).mean()

### the ones that survived have up to 4 siblings, and with 3 siblings you actually spend the highest amount of money
##only women with 3 siblings survived

titanic_edl[["Survived","SibSp","Pclass","Fare"]].groupby(["Survived","SibSp","Pclass"]).mean()
###fare price for 3 siblings is the same in survived and non survived--> it only matters if you are a female in oredr to survive

dist_sib_fare=titanic_edl[["Survived","SibSp","Pclass","Fare"]].groupby(["Survived","SibSp","Pclass"]).mean()
dist_sib_fare.reset_index(inplace=True)

plt.figure(figsize=(5,5))
sn.boxplot(x="SibSp",y="Fare",hue="Survived",data=dist_sib_fare)
plt.xlabel("Siblings")
plt.ylabel("Fare price")
plt.title("Fare prices based on #siblings")
plt.show()


###fare price and gender distribution
sex_sib=titanic_edl[["Survived","SibSp","Sex","Fare"]].groupby(["Survived","SibSp","Sex"]).mean()
sex_sib.reset_index(inplace=True)

plt.figure(figsize=(5,5))
sn.boxplot(x="Sex",y="Fare",hue="Survived",data=sex_sib)
plt.xlabel("Siblings")
plt.ylabel("Fare price")
plt.title("Fare prices based on #siblings")
plt.show()


####  let's look at the significance of name titles to survived class

titanic_edl.groupby("Title")["Survived"].aggregate(["mean","count"])

total_pop=titanic_edl["Survived"].count()
def weighted_survival(df):
    weight=df["Survived"].count()/total_pop
    surv=df["Survived"] * weight*100
    return np.sum(surv)

titanic_edl.groupby("Title").apply(weighted_survival).plot.bar()
plt.title("Avg. weighted Suvival rate by title (adj. by population size")
plt.ylabel("Survival rate in %")

titanic_edl.groupby(["Title","Pclass"])["Survived"].mean().unstack().plot.bar()
plt.title("Avg. weighted Suvival rate by title and class(adj. by population size")
plt.ylabel("Survival rate in %")

###let's investigate family size alone

titanic_edl.groupby(["Family_size"])["Survived"].mean().plot.bar()
plt.title("Survival by family size ")
plt.ylabel("Survival rate in %")

###let's investigate family size based on ther factors: gender, class

titanic_edl.groupby(["Family_size","Pclass"])["Survived"].mean().unstack().plot.bar()
plt.title("Survival by family size and class")
plt.ylabel("Survival rate in %")

####is survival rate dependent on family size and sex?

titanic_edl.groupby(["Family_size","Sex"])["Survived"].mean().unstack().plot.bar()
plt.title("Survival by family size and class")
plt.ylabel("Survival rate in %")



### whats the undelying distribution of male/ females to family size
titanic_edl.groupby(["Family_size","Sex"])["PassengerId"].count().unstack().plot.bar()
plt.title("Survival by family size and class")
plt.ylabel("Number of passengers")
plt.show()


###let"s look at parent alone
titanic_edl.groupby(["Parch"])["Survived"].mean().plot.bar(legend=False)
plt.title("Survival by direct dependecy: parents")
plt.ylabel("Survval rate")
plt.show()
###above depedence of 3 there are no survivers
####Parch dosen"t seem to add any other value

###parents by direct dependency
titanic_edl.groupby(["Parch","Sex"])["Survived"].mean().unstack().plot.bar()
plt.title("Survival by direct dependecy: parents")
plt.ylabel("Survval rate")
plt.show()

###make a dummy variable that encodes having siblings >4 !! 
# (the more dependency you have the less likly it is that you survive)
titanic_edl.groupby(["SibSp"])["Survived"].mean().plot.bar()
plt.title("Survival rate by direct dependency: child or spouse")
plt.ylabel("Survval rate")
plt.show()

### children dependent on gender
titanic_edl.groupby(["SibSp","Sex"])["Survived"].mean().unstack().plot.bar()
plt.title("Survival rate by gender and direct dependency: child or spouse")
plt.ylabel("Survval rate")
plt.show()

#####Let"s investigate if cabin is relevant for survival rate
titanic_edl["Level_cabin"].unique()

titanic_edl.groupby("Level_cabin")["Survived"].mean().plot.bar()
plt.title("Survival rates by cabin levels")
plt.ylabel("Survival rate")

###inspect cabin levels by class
titanic_edl.groupby(["Level_cabin","Pclass"])["Survived"].mean().unstack().plot.bar()
plt.title("Survival rates by cabin levels and class")
plt.ylabel("Survival rate")


####most of the upper level cabins belong to the 1st class--? there is a correlation between classse
# and cabins

titanic_edl.groupby(["Level_cabin","Pclass"])["Survived"].count().unstack().plot.bar()
plt.title("Survival rates by cabin levels and class")
plt.ylabel("#Passengers")

###bin split the data

titanic_edl["fam_size"]=pd.cut(titanic_edl["Family_size"], bins=3, labels=["small_fam","medium_fam","large_fam"])
titanic_edl["Sib_Sp_num"]=pd.cut(titanic_edl["SibSp"], bins=2, labels=["less_4","over_4"])

###heatmap with initial variables

one_hot_family=pd.get_dummies(titanic_edl["fam_size"])
one_hot_sibling=pd.get_dummies(titanic_edl["Sib_Sp_num"])
one_hot_sex=pd.get_dummies(titanic_edl["Sex"])
one_hot_title=pd.get_dummies(titanic_edl["Title"])

#with titles
# corr1=pd.concat([titanic_edl,one_hot_family,one_hot_sibling,one_hot_sex,one_hot_title],axis=1)

#without titles
corr1=pd.concat([titanic_edl,one_hot_family,one_hot_sibling,one_hot_sex],axis=1)

corr1.drop(["PassengerId","Family_size","Parch","SibSp","over_4","male", "medium_fam"],axis=1,inplace=True)
plt.figure(figsize=(10,10))
sn.heatmap(corr1.corr()[['Survived']],cmap="RdBu_r",center=0.0, annot=True)

