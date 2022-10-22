import streamlit as st

from sklearn.metrics import mean_squared_error
from sklearn import datasets,ensemble
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


import numpy as np

siteHeader = st.container()
modelTraining = st.container()
#customs changes can made here, e.g. chaning the background color, fontsize,etc
st.markdown(
      """
      <style>
      .main{
      background-color: #BDB76B;
      }
      <style>
      """,
      unsafe_allow_html=True
  )

def selection_of_params(col, keys):
    "Makes silder and boxes which allows an individual mix of variables"
    max_depth = col.slider('What should be the max_depth of the model?', min_value=5, max_value=50, 
    value=10, step=1, key = keys[0])
    
    number_of_trees = col.slider('How many trees should there be?', min_value=20, max_value=1000, 
    value=50, step=10, key= keys[1])
    
    learning_rate = col.selectbox('What should be the learning rate?', options=[0.0001,0.001,0.01,0.1], 
    index=2, key= keys[2])
    loss = 'squared_error'
    return max_depth,number_of_trees, learning_rate, loss

def model(params, col):
    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(X_train, y_train)

    mse_train = mean_squared_error(y_train, reg.predict(X_train))
    mse_test = mean_squared_error(y_test, reg.predict(X_test))
   
    col.write("Results for train: ")
    col.write("The mean squared error (MSE) on train set: {:.4f}".format(mse_train)) 
    col.write("Results for test: ")
    col.write("The mean squared error (MSE) on test set: {:.4f}".format(mse_test)) 
    return reg


with siteHeader:
    st.title("Welcome to my small AI project! :)")
    st.text('In this project I look into the diabetic dataset from sklearn.')
    st.text('The model used is a Gradient Boosting Regression Tree.')
    st.text('In the model training it is possible to compare to different choice of parameters.')

with modelTraining:
    st.header('Model training')
    #loads the dataset from sklearn 
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=13)
    #makes two colmnus to compare the result of different parameters 
    left_col, right_col =  st.columns(2)
    #gets the selected parameters for the left column
    max_depth,number_of_trees, learning_rate, loss = selection_of_params(left_col,[1,2,3,4])
    params_left = {"n_estimators": number_of_trees,"max_depth": max_depth,"min_samples_split": 5,
    "learning_rate": learning_rate,"loss": loss}
    #feeds the paraemeters into model, trains it and evaluate the test dataset
    # for the left column 
    reg_left = model(params_left, left_col)

    #gets the selected parameters for the right column
    max_depth,number_of_trees, learning_rate, loss = selection_of_params(right_col, [5,6,7,8])
    params_right = {"n_estimators": number_of_trees,"max_depth": max_depth,"min_samples_split": 5,
    "learning_rate": learning_rate,"loss": loss}
    #feeds the paraemeters into model, trains it and evaluate the test dataset
    # for the left colum
    reg_right = model(params_right ,right_col )


    

   
    
