import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

train_data = pd.read_csv('house_pred.csv')
X = train_data.drop(['Price'], axis=1)
y = train_data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = X.select_dtypes(include=[np.number]).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = numeric_transformer
X_train_processed = preprocessor.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_processed, y_train)

X_test_processed = preprocessor.transform(X_test)
y_pred = model.predict(X_test_processed)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', rmse)
