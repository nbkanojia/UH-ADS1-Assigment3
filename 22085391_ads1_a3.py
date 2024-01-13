# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 01:02:17 2024

@author: Nisarg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp
from sklearn.cluster import KMeans
import cluster_tools as ct
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import sklearn.preprocessing as pp
import scipy.optimize as opt


def read_world_bank_csv(filename):
    """
    Accept the csv filename with worldbank data format.
    Read a file and processes and prepare two dataframes by yeas as index
    and country as an index.

    Parameters
    ----------
    filename : string
        input the csv file name..

    Returns
    -------
    df_year_index : pandas.DataFrame
        DataFrame with years as an index.
    df_country_index : pandas.DataFrame
        DataFrame with the country as an index.

    """
    # set year range and country list to filter the dataset
    start_from_yeart = 1990
    end_to_year = 2021
    countrie_list = ["Brazil", "Indonesia", "Russian Federation", "Argentina",
                     "Paraguay", "Bolivia", "Nigeria"]

    # read csv using pandas
    wb_df = pd.read_csv(filename,
                        skiprows=3, iterator=False)

    # prepare a column list to select from the dataset
    years_column_list = np.arange(
        start_from_yeart, (end_to_year+1)).astype(str)
    all_cols_list = ["Country Name"] + list(years_column_list)
    
    
    # filter data: select only specific countries and years
    df_country_index = wb_df.loc[
        #wb_df["Country Name"].isin(countrie_list),
       :, all_cols_list]
    
    
    
    # make the country as index and then drop column as it becomes index
    df_country_index.index = df_country_index["Country Name"]
    df_country_index.drop("Country Name", axis=1, inplace=True)

    # convert year columns as interger
    df_country_index.columns = df_country_index.columns.astype(int)

    # clean na data, remove columns
    df_country_index.dropna(axis=0, inplace=True)   
    
    # Transpose dataframe and make the country as an index
    df_year_index = pd.DataFrame.transpose(df_country_index)


    # return the two dataframes year as index and country as index
    return df_year_index, df_country_index

def one_silhoutte(xy, n):
    """ Calculates silhoutte score for n clusters """

    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(xy)     # fit done on x,y pairs

    labels = kmeans.labels_
    
    # calculate the silhoutte score
    score = (skmet.silhouette_score(xy, labels))

    return score



def get_error_estimates(x, y, degree):
    """
   Calculates the error estimates of a polynomial function.
       """

    coefficients = np.polyfit(x, y, degree)
    y_estimate = np.polyval(coefficients, x)
    residuals = y - y_estimate

    return np.std(residuals)
def find_cluster(fert_data_cw,TFRT_data_cw):
    year=1990
    df_1990 = pd.DataFrame(index=fert_data_cw.index.copy())
    df_1990["Fertilizer Consumption"] = fert_data_cw.loc[:,year].copy()
    df_1990["Fertility rate, total (births per woman)"] = TFRT_data_cw.loc[:,year].copy() 
    
    
    year=2021
    df_2021 = pd.DataFrame(index=fert_data_cw.index.copy())
    df_2021["Fertilizer Consumption"] = fert_data_cw.loc[:,year].copy()
    df_2021["Fertility rate, total (births per woman)"] = TFRT_data_cw.loc[:,year].copy() 
    
    # visualising data
    df_norm_1990, df_min_1990, df_max_1990 = ct.scaler(df_1990)
    
    ## setup a scaler object
    #scaler = pp.RobustScaler()
    
    #for ic in range(2, 11):
    #     score = one_silhoutte(df_1990, ic)
     #    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")   # allow for minus signs
        
    
    ncluster = 2
    # set up the clusterer with the number of expected clusters
    kmeans_1990 = cluster.KMeans(n_clusters=ncluster, n_init=20)
    # Fit the data, results are stored in the kmeans object
    kmeans_1990.fit(df_norm_1990) # fit done on x,y pairs
    
    labels_1990 = kmeans_1990.labels_
    # extract the estimated cluster centres and convert to original scales
    cen_1990 = kmeans_1990.cluster_centers_
    
    xkmeans_1990 = cen_1990[:, 0]
    ykmeans_1990 = cen_1990[:, 1]
    # extract x and y values of data points
    x = df_norm_1990["Fertilizer Consumption"]
    y = df_norm_1990["Fertility rate, total (births per woman)"]
    plt.figure(figsize=(8.0, 8.0),dpi=300)
    # plot data with kmeans cluster number
    cm = plt.colormaps["Paired"]
    #plt.scatter(x, y, 10, labels_1990, marker="o", cmap=cm)
    # show cluster centres
    plt.style.use('seaborn')
    # visualising data
    fig, axs = plt.subplots(1, 2, figsize=(10, 5),dpi=300)
    axs[0].scatter(x, y, 10, labels_1990, marker="o", cmap=cm)
    axs[0].scatter(xkmeans_1990, ykmeans_1990, 45, "k", marker="d")
    axs[0].scatter(xkmeans_1990, ykmeans_1990, 45, "y", marker="+")
    axs[0].set_title('1990')
    axs[0].set_xlabel('Fertilizer Consumption')
    axs[0].set_ylabel('Fertility rate, total (births per woman)')
    
    
    
    
    df_norm_2021, df_min_2021, df_max_2021 = ct.scaler(df_2021)
    # set up the clusterer with the number of expected clusters
    kmeans_2021 = cluster.KMeans(n_clusters=ncluster, n_init=20)
    # Fit the data, results are stored in the kmeans object
    kmeans_2021.fit(df_norm_2021) # fit done on x,y pairs
    
    labels_2021 = kmeans_2021.labels_
    # extract the estimated cluster centres and convert to original scales
    cen_2021 = kmeans_2021.cluster_centers_
    
    xkmeans_2021 = cen_2021[:, 0]
    ykmeans_2021 = cen_2021[:, 1]
    # extract x and y values of data points
    x = df_norm_2021["Fertilizer Consumption"]
    y = df_norm_2021["Fertility rate, total (births per woman)"]
    
    # plot data with kmeans cluster number
    #cm = plt.colormaps["Paired"]
    #plt.scatter(x, y, 10, labels_2021, marker="o", cmap=cm)
    
    axs[1].scatter(x, y, 10, labels_2021, marker="o", cmap=cm )
    axs[1].scatter(xkmeans_2021, ykmeans_2021, 45, "k", marker="d")
    axs[1].scatter(xkmeans_2021, ykmeans_2021, 45, "y", marker="+")
    axs[1].set_title('2021')
    axs[1].set_xlabel('Fertilizer Consumption')
    axs[1].set_ylabel('Fertility rate, total (births per woman)')
    plt.tight_layout()
    plt.legend()
    
    #############################################
    
    
    
def poly2(x, a, b, c):
    """
    Calculates the value of a polynomial function of the form ax^2 + bx + c.

    """
    
    return a*x**2 + b*x + c

def poly3(x, a, b, c, d):
    """ Calulates polynominal"""
    
    x = x - 1990
    f = a + b*x + c*x**2 + d*x**3
    return f
def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    
    # makes it easier to get a guess for initial parameters
    t = t - 1990
    
    f = n0 * np.exp(g*t)
    
    return f

def poly4(x, a, b, c, d, e):
    """ Calulates polynominal"""
    
    x = x - 1990
    f = a + b*x + c*x**2 + d*x**3 + e*x**4
    
    return f
def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    
    f = n0 / (1 + np.exp(-g*(t - t0)))
    
    return f

    
def fitting(TFRT_data_yw,country):
    
    # fit data for Burundi
    df_fitting = TFRT_data_yw[[country]].apply(pd.to_numeric, errors='coerce')
    print(df_fitting.values)
    # Forecast for the next 20 years
    year = np.arange(1990, 2041)
    
    # fits the linear data
    param_b, cov_b = opt.curve_fit(poly4, df_fitting.index,
                                   df_fitting[country])
    
    # calculate standard deviation
    sigma_b = np.sqrt(np.diag(cov_b))
    
    # creates a new column for the fit figures
    df_fitting['fit'] = poly4(df_fitting.index, *param_b)
    
    # forecasting the fit figures
    forecast_b = poly4(year, *param_b)
    
    # error estimates
    error_b = get_error_estimates(df_fitting[country], df_fitting['fit'], 2)
    print('\n Error Estimates for Burundi GDP/Capita:\n', error_b)
    
    # Plotting the fit
    plt.style.use('seaborn')
    plt.figure(dpi=300)
    plt.plot(df_fitting.index, df_fitting[country],
             label="GDP/Capita", c='purple')
    plt.plot(year, forecast_b, label="Forecast", c='red')

    plt.xlabel("Year", fontweight='bold', fontsize=14)
    plt.ylabel("Fertility rate, total (births per woman)", fontweight='bold', fontsize=14)
    plt.legend()
    plt.title(country, fontweight='bold', fontsize=14)

    
    
    print(df_fitting)
    
########## Main ##########################

#Fertilizer consumption

fert_data_yw, fert_data_cw  = read_world_bank_csv("API_AG.CON.FERT.ZS_DS2_en_csv_v2_6305172.csv")
#Fertility rate
TFRT_data_yw, TFRT_data_cw  = read_world_bank_csv("API_SP.DYN.TFRT.IN_DS2_EN_csv_v2_6299995.csv")
#print(TFRT_data_yw)



#find_cluster(fert_data_cw,TFRT_data_cw)
############### Fiting
fitting(TFRT_data_yw,"World")


plt.show()









