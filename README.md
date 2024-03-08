# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly. Although neural networks are complex and computationally expensive, they are flexible and can dynamically pick the best type of regression, and if that is not enough, hidden layers can be added to improve prediction.

First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![image](https://github.com/NivethaKumar30/basic-nn-model/assets/119559844/79f70648-c996-4ee8-bf3f-53b4f18bdfe0)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:NIVETHA
### Register Number:212222230102

Dependencies:
```
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
```

Data From Sheets:
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('data').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
```
Data Visualization:
```
df = df.astype({'input':'float'})
df = df.astype({'output':'float'})
df
x=df[['input']].values
y=df[['output']].values
```
Data split and Preprocessing:
```
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train1=scaler.transform(x_train)
```
Regressive Model:
```
ai_brain = Sequential([
    Dense(6,activation = 'relu'),
    Dense(6,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(x_train1,y_train,epochs  = 2000)
```
Loss Calculation:
```
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
```
Evaluate the model:
```
x_test1 = scaler.transform(x_test)
ai_brain.evaluate(x_test1,y_test)
```
Prediction:
```
x_n1 = [[5]]
x_n1_1 = scaler.transform(x_n1)
ai_brain.predict(x_n1_1)
```
## Dataset Information

![Screenshot 2024-03-08 232910](https://github.com/NivethaKumar30/basic-nn-model/assets/119559844/e01a3186-57bb-4635-866f-2f5b70b0eef0)


## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2024-03-08 232321](https://github.com/NivethaKumar30/basic-nn-model/assets/119559844/1145c902-a25f-4e97-bb4a-9f530f88bbdf)

Training

![Screenshot 2024-03-08 232643](https://github.com/NivethaKumar30/basic-nn-model/assets/119559844/a94c472b-d09d-4490-9ec1-2ec2e20b3249)


### Test Data Root Mean Squared Error

![Screenshot 2024-03-08 232540](https://github.com/NivethaKumar30/basic-nn-model/assets/119559844/66c471eb-38ca-41ff-b684-087b63895d6f)


### New Sample Data Prediction

![Screenshot 2024-03-08 232556](https://github.com/NivethaKumar30/basic-nn-model/assets/119559844/35477db7-24a2-46ee-8e8e-c125f88d36cd)


## RESULT

A neural network regression model for the given dataset has been developed Sucessfully.
