# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 20:42:19 2024

@author: Nisarg
"""

import numpy as np
import pandas as pd
import cluster_tools as ct
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import sklearn.preprocessing as pp
import scipy.optimize as opt
import errors as err

def read_world_bank_csv(filename):

    # set year range and country list to filter the dataset
    start_from_yeart = 1990
    end_to_year = 2021
    countrie_list = ["High income", "Low income",  #["Brazil", "Indonesia", "Russian Federation", "Argentina","Paraguay", "Bolivia", "Nigeria","India",
                     "World"]

    # read csv using pandas
    wb_df = pd.read_csv(filename,
                        skiprows=3, iterator=False)

    # clean na data, remove columns
    wb_df.dropna(axis=1)

    # prepare a column list to select from the dataset
    years_column_list = np.arange(
        start_from_yeart, (end_to_year+1)).astype(str)
    all_cols_list = ["Country Name"] + list(years_column_list)

    # filter data: select only specific countries and years
    df_country_index = wb_df.loc[\
       # wb_df["Country Name"].isin(countrie_list),
     :,all_cols_list]

    # make the country as index and then drop column as it becomes index
    df_country_index.index = df_country_index["Country Name"]
    df_country_index.drop("Country Name", axis=1, inplace=True)

    # convert year columns as interger
    df_country_index.columns = df_country_index.columns.astype(int)

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


def poly(x, a, b, c):
    """ Calulates polynominal"""
    
    x = x - 1990
    f = a + b*x + c*x**2 #+ d*x**3# + e*x**4
    
    return f


def find_cluster(df_cluster,selected_column_1,selected_column_2, title):
    

    ###############Clustering   ######################################


    df_norm, df_min, df_max = ct.scaler(df_cluster)


    # calculate silhouette score for 2 to 10 clusters
    #for ic in range(2, 11):
    #    score = one_silhoutte(df_cluster, ic)
    #    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")   # allow for minus signs
    

    ncluster = 3


    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster, n_init=20)
    # Fit the data, results are stored in the kmeans object
    cluster_fit = kmeans.fit_predict(df_norm) # fit done on x,y pairs
    df_cluster["cluster"] =cluster_fit
    #print( df_cluster)
    labels = kmeans.labels_
  
    # extract the estimated cluster centres and convert to original scales
    cen = kmeans.cluster_centers_
    #cen = scaler.inverse_transform(cen)
    cen = ct.backscale(cen, df_min, df_max)
    
    xkmeans = cen[:, 0]
    ykmeans = cen[:, 1]

    # extract x and y values of data points
    
    plt.figure(figsize=(8.0, 8.0),dpi=300)
    # plot data with kmeans cluster number
    cm = plt.colormaps["Set1"]
    colors = ['red', 'green', 'blue', 'orange', 'purple']
   # plt.scatter(x, y, 80, labels, marker="o", cmap=cm)
    for label in np.unique(labels):
        x = df_cluster.loc[df_cluster["cluster"]==label][selected_column_1]
        y = df_cluster.loc[df_cluster["cluster"]==label][selected_column_2]
        plt.scatter(x, y, label=f'Cluster {label}', color=colors[label], 
                    edgecolors='k', s=100,alpha=0.5)
    
    
    #sc = plt.scatter(x, y, c=[colors[label] for label in labels], edgecolors=[colors[label] for label in labels], s=100, alpha=0.5)
    # show cluster centres
    #plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
    plt.scatter(xkmeans, ykmeans, 80, "y", marker="X" , label="Cluster Center")
    plt.xlabel(selected_column_1, fontsize=24, color='black')
    plt.ylabel(selected_column_2, fontsize=24, color='black')
    
    # X-axis Tick Labels Font Size
    plt.tick_params(axis='x', labelsize=18)

    # Y-axis Tick Labels Font Size
    plt.tick_params(axis='y', labelsize=18)

    plt.title(title, fontsize=30, color='navy')
    # Remove axis for a cleaner look
    #plt.axis('off')
    plt.legend()

def fitting_forcast(df_cluster,selected_column,title,forcast_to_year):
    
    #####################fitting############################


    #plt.figure()
    df_cluster["Year"] = df_cluster.index


    param, covar = opt.curve_fit(poly, df_cluster["Year"], df_cluster[selected_column])
    #df_cluster["fit"] = poly(df_cluster["Year"], *param)

    #df_cluster.plot("Year", ["co2", "fit"])


    ############Forcast###############
    year = np.arange(1990, forcast_to_year)
    forecast = poly(year, *param)
    sigma = err.error_prop(year, poly, param, covar)

    low = forecast - sigma
    up = forecast + sigma

    df_cluster["fit"] = poly(df_cluster["Year"], *param)
    
    plt.figure(figsize=(10.0, 8.0),dpi=300)
    plt.plot(df_cluster["Year"], df_cluster[selected_column], label=selected_column, linewidth=3)
    plt.plot(year, forecast, label="forecast" , linestyle=":" , linewidth=2.5)

    # plot uncertainty range
    plt.fill_between(year, low, up, color='Orange', alpha=0.4, label="fit")
    plt.ylim(0,20)
    plt.xlabel("Year", fontsize=24, color='black')
    plt.ylabel("Death rate, crude (per 1,000 people)", fontsize=24, color='black')
    
    # X-axis Tick Labels Font Size
    plt.tick_params(axis='x', labelsize=18)

    # Y-axis Tick Labels Font Size
    plt.tick_params(axis='y', labelsize=18)

    plt.title(title, fontsize=30, color='navy')
    plt.legend()

###### Main Function ################

# read csv files and get the dataframs

co2_data_yw, co2_data_cw = \
    read_world_bank_csv("API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5994970.csv")

agri_lnd_yw, agri_lnd_cw = \
    read_world_bank_csv("API_AG.LND.AGRI.ZS_DS2_en_csv_v2_5995314.csv")
    

fert_data_yw, fert_data_cw  = read_world_bank_csv("API_AG.CON.FERT.ZS_DS2_en_csv_v2_6305172.csv")

#Fertility rate
TFRT_data_yw, TFRT_data_cw  = read_world_bank_csv("API_SP.DYN.TFRT.IN_DS2_EN_csv_v2_6299995.csv")


#Death rate, crude (per 1,000 people)
death_data_yw, death_data_cw  = read_world_bank_csv("API_SP.DYN.CDRT.IN_DS2_en_csv_v2_6303594.csv")

#Birth rate, crude (per 1,000 people)
birth_data_yw, birth_data_cw  = read_world_bank_csv("API_SP.DYN.CBRT.IN_DS2_en_csv_v2_6301675.csv")

df_cluster = pd.DataFrame()
df_cluster["High income"] = death_data_yw["High income"]
df_cluster["Low income"] = death_data_yw["Low income"]  #birth_data_yw["Low income"]

find_cluster(df_cluster,"High income","Low income","Death rate, crude (per 1,000 people)")
fitting_forcast(df_cluster,"High income","High Income",2025)
fitting_forcast(df_cluster,"Low income","Low Income",2025)
fitting_forcast(death_data_yw,"Afghanistan","Low Income(Afghanistan)",2025)
# show all plots
plt.show()
