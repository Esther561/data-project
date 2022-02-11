# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 05:32:03 2020

@author: 启超
"""

import pandas as pd
import seaborn as sns
#import the necessary module
from sklearn import preprocessing
import seaborn as sb


cases_data = pd.read_csv("G:/我的云端硬盘/semester2/722/assignment/2/COVID19 cases Toronto.csv")


# Describe the Data
print(cases_data)


print(cases_data.head(2))
print(cases_data.tail(4))
print(cases_data.dtypes)

# Explore data

# sns.set(rc={'figure.figsize':(16,13)})
# sns.countplot(cases_data['Outcome'])


# sns.set(style='dark',color_codes=True)
# sns.set(rc={'figure.figsize':(11.7,8)})
# sns.countplot('Age Group', data=cases_data, hue='Outcome')
# sns.countplot('Outbreak Associated', data=cases_data, hue='Outcome')
# sns.countplot('Source of Infection', data=cases_data, hue='Outcome')
# sns.countplot('Currently Hospitalized ', data=cases_data, hue='Outcome')
# sns.countplot('Currently in ICU', data=cases_data, hue='Outcome')
# sns.countplot('Currently Intubated', data=cases_data, hue='Outcome')
# sns.countplot('Ever Hospitalized', data=cases_data, hue='Outcome')
# sns.countplot('Ever in ICU', data=cases_data, hue='Outcome')
# sns.countplot('Ever Intubated', data=cases_data, hue='Outcome')
# sns.countplot('Client Gender', data=cases_data, hue='Outcome')






# Verify the data quality
nulldata=cases_data.isnull().sum()
print(nulldata)


detection=cases_data.describe()
print(detection)




# select data

select_data=cases_data[['Outbreak Associated',
                        'Age Group','Client Gender','Source of Infection',
                        'Outcome','Currently Hospitalized',
                        'Currently in ICU','Currently Intubated',
                        'Ever Hospitalized','Ever in ICU',
                        'Ever Intubated']]
print(select_data.head)


nulldata=select_data.isnull().sum()
print(nulldata)


select_data['Age Group'].fillna(method = 'pad',inplace = True)
nulldata=select_data.isnull().sum()
print(nulldata)

# Construct the data
frame = pd.DataFrame(select_data, columns=['Outbreak Associated',
                        'Age Group','Client Gender','Source of Infection',
                        'Outcome','Currently Hospitalized',
                        'Currently in ICU','Currently Intubated',
                        'Ever Hospitalized','Ever in ICU',
                        'Ever Intubated'])
  
def function(a, b):
    if a == 'Yes' or b == 'Yes':
        return 'Yes'
    else:
        return 'No'
 
frame['Hospitalized'] = frame.apply(lambda x: function(x["Currently Hospitalized"], x["Ever Hospitalized"]), axis=1)

 
frame['ICU'] = frame.apply(lambda x: function(x["Currently in ICU"], x["Ever in ICU"]), axis=1)

frame['Intubated'] = frame.apply(lambda x: function(x["Currently Intubated"], x["Ever Intubated"]), axis=1)

frame['Outcome2']=frame['Outcome']

frame.loc[frame['Outcome'] =='ACTIVE','Outcome2']='NONFATAL'
frame.loc[frame['Outcome'] =='RESOLVED','Outcome2']='NONFATAL'


frame.loc[frame['Client Gender'] =='UNKNOWN','Client Gender']='OTHER'


print(frame)





# Format the data as required
# create the Labelencoder object
le = preprocessing.LabelEncoder()

print("Outbreak Associated: ",frame['Outbreak Associated'].unique())
print("Source of Infection: ",frame['Source of Infection'].unique())
print("Age Group : ",frame['Age Group'].unique())
print("Outcome2 : ",frame['Outcome2'].unique())
print("ICU: ",frame['ICU'].unique())
print("Intubated: ",frame['Intubated'].unique())
print("Hospitalized: ",frame['Hospitalized'].unique())
print("Client Gender: ",frame['Client Gender'].unique())



# convert the categorical columns into numeric
frame['Outbreak Associated'] = le.fit_transform(frame['Outbreak Associated'])
frame['Source of Infection'] = le.fit_transform(frame['Source of Infection'])
frame['Age Group'] = le.fit_transform(frame['Age Group'].astype(str))
frame['Outcome2'] = le.fit_transform(frame['Outcome2'])
frame['ICU'] = le.fit_transform(frame['ICU'])
frame['Intubated'] = le.fit_transform(frame['Intubated'])
frame['Hospitalized'] = le.fit_transform(frame['Hospitalized']) 
frame['Client Gender'] = le.fit_transform(frame['Client Gender'])
                                                   

#display the initial records
print(frame.head())


frame=frame.drop(['Currently Hospitalized','Outbreak Associated',
                        'Currently in ICU','Currently Intubated',
                        'Ever Hospitalized','Ever in ICU',
                        'Ever Intubated','ICU','Outcome'],axis=1)
print("column_name:",frame.columns)

# 4.1 Reduce the Data
X = frame.iloc[:,0:4]  
y = frame.iloc[:,-1]
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
#use inbuilt class feature_importances of tree based classifiers
print(model.feature_importances_)
#plot graph of feature importances for better visualization

# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(4).plot(kind='barh')
# plt.show()




frame=frame.drop(['Intubated'],axis=1)
print("column_name:",frame.columns)






# 4.2 Project the Data
# sns.countplot('Outcome2', data=frame)
print(frame['Outcome2'].value_counts())
print("-------------------------")
print(frame['Outcome2'].value_counts(normalize=True))




df1=frame[frame['Outcome2']==1]
df0=frame[frame['Outcome2']==0]

df2=df1.sample(frac=0.09)

df_new=pd.concat([df0,df2])

x=df_new.iloc[:,1:-1]
y=df_new["Outcome2"]

# sns.countplot('Outcome2', data=df_new)
print(y.value_counts())
print("-------------------------")
print(y.value_counts(normalize=True))




#select columns other than 'Outcome2'
cols = [col for col in df_new.columns if col not in ['Outcome2']]

#dropping the 'Outcome2'column
data = df_new[cols]

#assigning the Oppurtunity Result column as target
target = df_new['Outcome2']

print(data.head(n=2))




# ct_counts1 = df_new.groupby(['Age Group','Outcome2']).size()
# ct_counts1 = ct_counts1.reset_index(name = 'count')
# ct_counts1 = ct_counts1.pivot(index ='Age Group', columns = 'Outcome2', values = 'count')
# sb.heatmap(ct_counts1)

# ct_counts2 = df_new.groupby(['Client Gender','Outcome2']).size()
# ct_counts2 = ct_counts2.reset_index(name = 'count')
# ct_counts2 = ct_counts2.pivot(index ='Client Gender', columns = 'Outcome2', values = 'count')
# sb.heatmap(ct_counts2)

# ct_counts3 = df_new.groupby(['Source of Infection','Outcome2']).size()
# ct_counts3 = ct_counts3.reset_index(name = 'count')
# ct_counts3 = ct_counts3.pivot(index ='Source of Infection', columns = 'Outcome2', values = 'count')
# sb.heatmap(ct_counts3)

# ct_counts4 = df_new.groupby(['Hospitalized','Outcome2']).size()
# ct_counts4 = ct_counts4.reset_index(name = 'count')
# ct_counts4 = ct_counts4.pivot(index ='Hospitalized', columns = 'Outcome2', values = 'count')
# sb.heatmap(ct_counts4)









#import the necessary module
from sklearn.model_selection import train_test_split

#split data set into train and test sets
data_train, data_test, target_train, target_test = train_test_split(data,target,
                                            test_size = 0.30, random_state = 10)





#import the necessary module
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#create an object of the type GaussianNB
gnb = GaussianNB()

#train the algorithm on training data and predict using the testing data
pred = gnb.fit(data_train, target_train).predict(data_test)


#print the accuracy score of the model
print("Naive-Bayes accuracy : ",accuracy_score(target_test, pred, normalize = True))




# from  yellowbrick.classifier import ClassificationReport
# # Instantiate the classification model and visualizer
# visualizer = ClassificationReport(gnb, classes=['NONFATAL','FATAL'])
# visualizer.fit(data_train, target_train) # Fit the training data to the visualizer
# visualizer.score(data_test, target_test) # Evaluate the model on the test data
# g = visualizer.poof() # Draw/show/poof the data











#import the necessary modules
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#create an object of type LinearSVC
svc_model = LinearSVC(random_state=0)

#train the algorithm on training data and predict using the testing data
pred = svc_model.fit(data_train, target_train).predict(data_test)

#print the accuracy score of the model
print("LinearSVC accuracy : ",accuracy_score(target_test, pred, normalize = True))




from yellowbrick.classifier import ClassificationReport

# Instantiate the classification model and visualizer
visualizer = ClassificationReport(svc_model, classes=['NONFATAL','FATAL'])

visualizer.fit(data_train, target_train) # Fit the training data to the visualizer
visualizer.score(data_test, target_test) # Evaluate the model on the test data
g = visualizer.poof() # Draw/show/poof the data



# 8.3
# print(" Age Group: ",frame['Age Group'].unique())
# frame['Age Group'] = le.fit_transform(frame['Age Group'].astype(str))
# print(" Age Group: ",frame['Age Group'].unique())
# sns.countplot('Age Group', data=df_new, hue='Outcome2')


# print("Source of Infection: ",frame['Source of Infection'].unique())
# frame['Source of Infection'] = le.fit_transform(frame['Source of Infection'])
# print("Source of Infection: ",frame['Source of Infection'].unique())
# sns.countplot('Source of Infection', data=df_new, hue='Outcome2')



# print("Hospitalized : ",frame['Hospitalized'].unique())
# frame['Hospitalized'] = le.fit_transform(frame['Hospitalized'])
# print("Hospitalized : ",frame['Hospitalized'].unique())
# sns.countplot('Hospitalized', data=df_new, hue='Outcome2')


# print("Client Gender : ",frame['Client Gender'].unique())
# frame['Client Gender'] = le.fit_transform(frame['Client Gender'])
# print("Client Gender : ",frame['Client Gender'].unique())
# sns.countplot('Client Gender', data=df_new, hue='Outcome2')















