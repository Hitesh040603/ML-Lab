#%% 1
import numpy as np
x=np.random.rand(1500).reshape(-1,1)
y=np.random.rand(1500).reshape(-1,1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,
test_size=500)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)



import matplotlib.pyplot as plt
plt.scatter(x_test,y_test)

y_pred = reg.predict(x_test)
plt.plot(x_test,y_pred,c='r')
error=np.mean(x_train)**2
y_pred = reg.predict(x_train)
error=np.mean(y_train - y_pred )**2
errors=[]
errors.append(error)
best_fit_index=np.argmin(errors)
reg.intercept_
error=np.mean(y_train-y_pred)**2
print(error)


#%% 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def regression(x,y):
    meanx=x.mean()
    print(meanx)
    meany=y.mean()
    print(meany)
    
    
    num=np.sum((x-meanx)*(y-meany))
    den=((np.sum((x-meanx)**2))*(np.sum((y-meany)**2)))**0.5
    r=num/den
    print(r)
    sx=np.std(x)
    sy=np.std(y)
    b1=r*(sy/sx)
    b0=meany-(b1*meanx)
    preds=b0+(b1*x)
    

    return preds
    


data=pd.read_csv(r'C:\Users\Hitesh\OneDrive\Desktop\SNU\Sem 6\ML Lab\Ex1\data1.csv')


x=data.iloc[:,0]
y=data.iloc[:,-1]
preds=regression(x,y)
plt.scatter(x,y)
plt.plot(x,preds)
