#!/usr/bin/env python3
import sys
import time
import math
import pandas as pd
import numpy as np
from numpy import linalg as LA
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score

''' develop the best predictive model based on the chemical engineering dataset'''

__author__ = 'Jacob Hajjar, Michael-Ken Okolo'
__email__ = 'hajjarj@csu.fullerton.edu, michaelken.okolo1@csu.fullerton.edu'
__maintainer__ = 'jacobhajjar, michaelkenokolo'


def main():
    """the main function"""
    data_frame = pd.read_csv("Data1.csv")
    #xy_data = data_frame.to_numpy()
    #xy_standardized_data = (xy_data - np.mean(xy_data)) / np.std(xy_data)
    #x_data = xy_standardized_data[:,[0,1,2,3]]
    #y_data = xy_standardized_data[:, 4]

    x_data = data_frame[["T", "P", "TC", "SV"]].to_numpy()
    y_data = data_frame["Idx"].to_numpy()

    #standardize data and estimate covariance matrix
    x_standardized_data = preprocessing.StandardScaler().fit_transform(x_data)
    #verify standardized data
    for index in range(0,4,1):
        print("{:.1f}".format(x_standardized_data[...,index].mean()))
        print("{:.1f}".format(np.std(x_standardized_data[...,index])))
    
    cov_1 = np.dot(x_data.T, x_data)
    cov_matrix = np.cov(x_standardized_data.T)
    print(cov_1)
    print(cov_matrix)
    #calculate the eigvenvalues and eigenvector
    
    # eig_val, eig_vec = LA.eig(cov_matrix)

    #find projection matrix

    #proj_matrix = np.dot(x_standardized_data.T, eig_vec)

    #component matrix loading



if __name__ == '__main__':
    main()
