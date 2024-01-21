# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 20:42:19 2024

@author: Nisarg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as skmet
import scipy.optimize as opt
from sklearn import cluster
import errors as err
import cluster_tools as ct


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
    start_from_yeart = 1960
    end_to_year = 2021

    # read csv using pandas
    wb_df = pd.read_csv(filename,
                        skiprows=3, iterator=False)

    # clean data, remove columns
    wb_df.dropna(axis=1)

    # prepare a column list to select from the dataset
    years_column_list = np.arange(
        start_from_yeart, (end_to_year+1)).astype(str)
    all_cols_list = ["Country Name"] + list(years_column_list)

    # filter data: select only specific countries and years
    df_country_index = wb_df.loc[\
        # wb_df["Country Name"].isin(countrie_list),
        :, all_cols_list]

    # make the country as index and then drop column as it becomes index
    df_country_index.index = df_country_index["Country Name"]
    df_country_index.drop("Country Name", axis=1, inplace=True)

    # convert year columns as interger
    df_country_index.columns = df_country_index.columns.astype(int)

    # Transpose dataframe and make the country as an index
    df_year_index = pd.DataFrame.transpose(df_country_index)

    # return the two dataframes year as index and country as index
    return df_year_index, df_country_index


def get_world_bank_metadata_country(filename, filter_income_group):
    """
    read the metadata csv and reutrn country list by imcome group.

    Parameters
    ----------
    filename : string
        metadata file name.
    filter_income_group : string
        income group filter name.

    Returns
    -------
    country_list : pandas.DataFrame
        list of filtered country.

    """
    wb_md_df = pd.read_csv(filename, iterator=False)
    country_list = wb_md_df.loc[wb_md_df["IncomeGroup"] ==
                                filter_income_group]["TableName"]
    return country_list


def one_silhoutte(df_xy, num_of_cluster):
    """
    Calculates silhoutte score for n clusters.

    Parameters
    ----------
    xy : pandas.DataFrame
        dataset for which need to finc kmean fit.
    n : int
        number of cluster.

    Returns
    -------
    score : TYPE
        DESCRIPTION.

    """

    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=num_of_cluster, n_init=20)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_xy)     # fit done on x,y pairs

    labels = kmeans.labels_

    # calculate the silhoutte score
    score = skmet.silhouette_score(df_xy, labels)

    return score


def poly_fit_function(var_x, const_a, const_b, const_c):  # , const_d):
    """ Calulates polynominal. """

    var_x = var_x - 1960
    poly_fun = const_a + const_b*var_x + const_c*var_x**2

    return poly_fun


def find_and_plot_cluster(df_cluster, selected_column_1, selected_column_2,
                          title, filename):
    """
    find the cluster in dataset and display on the graph.

    Parameters
    ----------
    df_cluster : pandas.DataFrame
        dateset for finding cluster.
    selected_column_1 : string
        seleced column 1 for comparesion.
    selected_column_2 : string
        seleced column 2 for comparesion.
    title : string
        title for graph.
    filename: string
        file name for save graph on disk.

    Returns
    -------
    None.

    """

    # normalize the data for clustring
    df_norm, df_min, df_max = ct.scaler(df_cluster)

    # calculate silhouette score for 2 to 10 clusters
    for ic in range(2, 11):
        score = one_silhoutte(df_cluster, ic)
        print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")

    ncluster = 3

    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster, n_init=20)
    # Fit the data, results are stored in the kmeans object
    cluster_fit = kmeans.fit_predict(df_norm)  # fit done on x,y pairs
    df_cluster["cluster"] = cluster_fit
    labels = kmeans.labels_

    # extract the estimated cluster centres and convert to original scales
    cen = kmeans.cluster_centers_
    # denormalize the cluster centers
    cen = ct.backscale(cen, df_min, df_max)

    plt.figure(figsize=(10.0, 8.0), dpi=300)

    # color for clusters
    colors = ['red', 'green', 'blue', 'orange', 'purple']

    # plot data with kmeans cluster number
    for label in np.unique(labels):
        # extract x and y values of data points
        df_xaxis = \
            df_cluster.loc[df_cluster["cluster"] == label][selected_column_1]
        df_yaxis = \
            df_cluster.loc[df_cluster["cluster"] == label][selected_column_2]
        plt.scatter(df_xaxis, df_yaxis, label=f'Cluster {label}',
                    color=colors[label],
                    edgecolors='k', s=100, alpha=0.5)

    # show cluster centres
    plt.scatter(cen[:, 0], cen[:, 1], 80, "y", marker="X",
                label="Cluster Center")
    plt.xlabel(selected_column_1, fontsize=26, color='black')
    plt.ylabel(selected_column_2, fontsize=26, color='black')

    # x-axis tick labels font size
    plt.tick_params(axis='x', labelsize=20)

    # y-axis tick labels font size
    plt.tick_params(axis='y', labelsize=20)

    # set graph labels title and limit
    plt.ylim(0, 60)

    plt.title(title, fontsize=30, color='#035bbc')

    plt.legend(fontsize=20)
    plt.grid(True)
    # save file
    plt.savefig(filename, dpi=300, bbox_inches='tight')


def fitting_and_forcast(df_cluster, selected_column, title, forcast_to_year,
                        ylim_max, filename):
    """
    find the the fit for the givin dataframe and using it make the forcast

    Parameters
    ----------
    df_cluster : pandas.DataFrame
        dataset to find the fitting and base on it make forcast.
    selected_column : string
        sleect the column from df_cluster for fitting and forcast.
    title : string
        title for .
    forcast_to_year : TYPE
        title for graph.
    filename: string
        file name for save graph on disk.

    Returns
    -------
    None.

    """
    # fitting

    df_cluster["Year"] = df_cluster.index
    # find the curve fit get param and co-veriables
    param, covar = opt.curve_fit(poly_fit_function, df_cluster["Year"],
                                 df_cluster[selected_column])

    # forcast

    # year range for forcast
    years = np.arange(df_cluster["Year"].min(), forcast_to_year)
    # call polynomial function to get forcast
    forecast = poly_fit_function(years, *param)
    # find the standard deviation
    sigma = err.error_prop(years, poly_fit_function, param, covar)
    low_bountry = forecast - sigma
    up_bountry = forecast + sigma

    # add the new forcast data into dataframe
    df_cluster["fit"] = poly_fit_function(df_cluster["Year"], *param)

    # plot the forcast on graph
    plt.figure(figsize=(10.0, 8.0), dpi=300)
    plt.plot(df_cluster["Year"], df_cluster[selected_column],
             label=selected_column, linewidth=3)
    plt.plot(years, forecast, label="forecast", linestyle=":", linewidth=2.5)

    # plot uncertainty range
    plt.fill_between(years, low_bountry, up_bountry, color='Orange',
                     alpha=0.4, label="fit")

    # x-axis tick labels font size
    plt.tick_params(axis='x', labelsize=20)
    # y-axis tick labels font size
    plt.tick_params(axis='y', labelsize=20)

    # set graph labels title and limit
    plt.ylim(0, ylim_max)
    plt.xlabel("Year", fontsize=26, color='black')
    plt.ylabel("Death rate, crude (per 1,000 people)", fontsize=26,
               color='black')
    plt.title(title, fontsize=30, color='#035bbc')
    plt.legend(fontsize=20)
    plt.grid(True)
    # save the graph image
    plt.savefig(filename, dpi=300, bbox_inches='tight')

########## Main Function ################
# read csv files and get the dataframs
# Death rate, crude (per 1,000 people)


death_data_yw, death_data_cw = \
    read_world_bank_csv("API_SP.DYN.CDRT.IN_DS2_en_csv_v2_6303594.csv")

# Birth rate, crude (per 1,000 people)
birth_data_yw, birth_data_cw = \
    read_world_bank_csv("API_SP.DYN.CBRT.IN_DS2_en_csv_v2_6301675.csv")


# prepare the date for clustring
df_for_cluster_1960 = pd.DataFrame(index=death_data_cw.index)
df_for_cluster_1960["death"] = death_data_cw[1960]
df_for_cluster_1960["birth"] = birth_data_cw[1960]
# clean data
df_for_cluster_1960.dropna(inplace=True)
find_and_plot_cluster(df_for_cluster_1960, "death", "birth",
                      "Death and Brirth rate, crude " +
                      "\n(per 1,000 people)(1960)", "cluster_1960.png")


# prepare the date for clustring
df_for_cluster_2021 = pd.DataFrame(index=death_data_cw.index)
df_for_cluster_2021["death"] = death_data_cw[2021]
df_for_cluster_2021["birth"] = birth_data_cw[2021]
# clean data
df_for_cluster_2021.dropna(inplace=True)
find_and_plot_cluster(df_for_cluster_2021, "death", "birth",
                      "Death and Brirth rate, crude " +
                      "\n(per 1,000 people)(2021)", "cluster_2021.png")


# forcast death
fitting_and_forcast(death_data_yw, "Nigeria",
                    "Nigeria, Death rate, crude \n(per 1,000 people)" +
                    "(Low Income Country)", 2030, 30, "nigeria_death.png")

# forcast death
fitting_and_forcast(death_data_yw, "United States",
                    "United States, Death rate, crude \n(per 1,000 people)" +
                    "(High Income Country)", 2030, 30, "us_death.png")

# forcast birth
fitting_and_forcast(birth_data_yw, "Nigeria",
                    "Nigeria, Birth rate, crude \n(per 1,000 people)" +
                    "(Low Income Country)", 2030, 60, "nigeria_birth.png")

# forcast birth
fitting_and_forcast(birth_data_yw, "United States",
                    "United States, Birth rate, crude \n(per 1,000 people)" +
                    "(High  Income Country)", 2030, 60, "us_birth.png")

# show all plots
plt.show()
