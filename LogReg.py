import pandas as pd
df1=pd.read_csv(r"C:\Users\Hitesh\OneDrive\Desktop\SNU\Sem 6\ML Lab\Ex3\telecom_customer_churn.csv")
df=df1.drop(['Customer ID','Zip Code','Latitude','Longitude','City','Churn Category','Churn Reason'],axis=1)



#%% assign values


df['Gender']=df['Gender'].map({'Male':1,'Female':0})
df['Married']=df['Married'].map({'Yes':1,'No':0})
df['Phone Service']=df['Phone Service'].map({'Yes':1,'No':0})
df['Multiple Lines']=df['Multiple Lines'].map({'Yes':1,'No':0})
df['Internet Service']=df['Internet Service'].map({'Yes':1,'No':0})

col=['Online Backup','Device Protection Plan','Premium Tech Support','Streaming TV','Streaming Movies','Streaming Music','Unlimited Data','Paperless Billing']

df['Online Security']=df['Online Security'].map({'Yes':1,'No':0})

for i in col:
    df[i]=df[i].map({'Yes':1,'No':0})
#%% dummies

S=pd.get_dummies(df['Payment Method'],dtype='int')
df=df.join(S)

S=pd.get_dummies(df['Internet Type'],dtype='int')
df=df.join(S)

S=pd.get_dummies(df['Contract'],dtype='int')
df=df.join(S)

df=df.drop(['Payment Method','Internet Type','Contract','Offer'],axis=1)


#%%null values
colnull=[
 'Avg Monthly Long Distance Charges',
 'Multiple Lines',
 'Avg Monthly GB Download',
 'Online Security',
 'Online Backup',
 'Device Protection Plan',
 'Premium Tech Support',
 'Streaming TV',
 'Streaming Movies',
 'Streaming Music',
 'Unlimited Data']

df=df.dropna(subset=colnull)


#%%dependent and independent

y=df.iloc[:,-10]
df.drop(['Customer Status'],axis=1)
x=df



#%%splitting

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=16)


#%%scaling
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x_train)
x_scaled = scaler.transform(x_train)
#%%training

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_scaled,y_train)


#%%predicting
y_pred = logreg.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))




