"""
Author: Samuel Oseh
Date: 9th April, 2021
Name of App: Credit Card Fraud Detector
Licence: MIT 

"""

# python libraries
import os
import pickle

# streamlit
import streamlit as st 

# data analysis libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

# sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

@st.cache()
def load_data(path):
    return pd.read_csv(path)

def predict(X):
    """Predict test data"""
    
    # load the model
    model = pickle.load(open('model/forest.pkl', 'rb'))

    prediction_prob = model.predict_proba(X)[:, 0]
    prediction = model.predict(X)

    return prediction, prediction_prob


def display_prediction(prediction, prediction_prob):
    if prediction[0] == 1:
        st.success('Normal Transaction with a {:.2f}% probabibity of being fraudulent'.format(prediction_prob[0] * 100))
    else: 
        st.warning('Fraudulent Transaction with a {:.2f}% probability of being fraudulent'.format(prediction_prob[0] * 100))

def main():
    """Credit Fraud Detection Application"""

    st.title('Credit Fraud Detection')
    menu = ['Home', 'Prediction']
    st.sidebar.markdown('# Menu')
    choice = st.sidebar.selectbox('', menu)

    PATH = 'data/creditcard.csv'
    data = load_data(PATH)

    if choice == 'Home':
        st.markdown('***Why Detect Fraudulent Transactions?***')
        st.markdown("Credit card fraud is on the increase as technology and global\nsuper highways develop.The cost to both businesses and consumers\nfrom this type of fraud costs billions of dollars every year.\nFraudsters are continually finding new ways to commit their illegal activities. \nAs a result, it has become essential for financial institutions and businesses to \ndevelop advanced fraud detection techniques to counter the \nthreat of fraudulent credit card transactions and identity theft and keep losses to a minimum.")
        
    if choice == 'Prediction':
        st.subheader('Predictive Analysis')

        feature_list = []

        columns = ['V3', 'V4', 'V10', 'V12', 'V14', 'V17']

        features_label = ['Feature {}'.format(i) for i in range(1, len(columns) + 1)]

        for col, col_name in zip(columns, features_label):
            min_value = data[col].min()
            max_value = data[col].max()
            feature_list.append(st.number_input(col_name, min_value, max_value))

        feature_list = np.array([feature_list])

        if st.button('Predict'):
            prediction, prediction_prob = predict(feature_list)
            display_prediction(prediction, prediction_prob)

if __name__ == "__main__":
    main()
