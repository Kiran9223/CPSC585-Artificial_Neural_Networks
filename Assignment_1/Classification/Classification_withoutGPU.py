from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
import time

cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
# Feature extraction
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets
# Print metadata
print(cdc_diabetes_health_indicators.metadata)

print(X.isnull().sum())

# Feature Scaling
scaler = StandardScaler()
X_scale = scaler.fit_transform(X)

X_scale = pd.DataFrame(X_scale, columns=X.columns)
print(X_scale.head())

# Train Test split
X_train,X_test,y_train,y_test=train_test_split(X_scale, y, test_size=0.2, random_state=26)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Model training on training data
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
start_time = time.time()
cpu_time = time.time() - start_time


# Prediction on test data
y_pred = log_reg.predict(X_test)

# Model performance evaluation
print("Confusion matrix\n",confusion_matrix(y_test,y_pred))
print("Accuracy score",accuracy_score(y_test,y_pred))
print("Precision score",precision_score(y_test,y_pred))
print("Computation Time {:.2f} seconds".format(cpu_time))

'''
/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py:1408: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
  y = column_or_1d(y, warn=True)
Confusion matrix
 [[42812   977]
 [ 5901  1046]]
Accuracy score 0.864435509303059
Precision score 0.5170538803756797
Computation Time: 0.00 seconds
'''