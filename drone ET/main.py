'''
date created 24 september 2024
by: Aakash Sharma
for sih Project

Main Function of this program is to create training model for a drone irrigation system so that efficient use of water
can be made while irrigation.
can be done and also to predict the water usage of the crops in the field.

current focus is to make the prototype for the idea
future plan is to make cost effective.

'''
#importing basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#importing sklearn modules in order to split the datasets and getting ready to predict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


#loading the datasets
data = pd.read_csv('D:\sih\drone ET\datasets\Daily_data_of_Soil_Moisture_during_April_2024.csv')

#checking for missing values
print(data.isnull())
#dropping the missing values
data.dropna(inplace=True)
print(data.isnull())

def load_data():
    data = pd.read_csv('D:\sih\drone ET\datasets\Daily_data_of_Soil_Moisture_during_April_2024.csv')
    return data
def data_preprocessing(data):
    data.drop('Avg_smlvl_at15cm', axis=1, inplace=True)
    data.drop('District', axis=1, inplace=True)
    data['Avg_smlvl_at15cm'] = data['Avg_smlvl_at15cm'].map({'M':1, 'B':0})
    return data
def split_data(data):
    X = data.drop('Avg_smlvl_at15cm', axis=1)
    y = data['Avg_smlvl_at15cm']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test
def feature_scaling(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test
def main():
    data = load_data()
    data = data_preprocessing(data)
    X_train, X_test = split_data(data)
    X_train, X_test = feature_scaling(X_train, X_test)
    return X_train, X_test
if __name__ == '__main__':
    X_train, X_test = main()
    print(X_train.shape, X_test.shape)
    print(X_train)
    print(X_test)
    print('Data Preprocessing Done')
