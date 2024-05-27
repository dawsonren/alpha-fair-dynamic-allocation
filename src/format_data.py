"""
Handle data from spreadsheet and format
"""

import pandas as pd

# import k-means clustering for geographic clustering of food pantries
import sklearn.cluster as cluster

# import matplotlib for plotting
import matplotlib.pyplot as plt


def clean_data(data):
    """
    Clean data from spreadsheet
    """
    # get selected columns
    data = data[
        [
            "Site Name",
            "latitude",
            "longitude",
            "Average Demand per Visit",
            "StDev(Demand per Visit)",
        ]
    ]
    # rename columns
    data.columns = ["site_name", "latitude", "longitude", "avg_demand", "std_demand"]
    return data


def cluster_data(data, n_clusters):
    """
    Cluster data into n_clusters
    """
    # get latitude and longitude
    X = data[["latitude", "longitude"]]
    # fit model
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=0, n_init=100)
    kmeans.fit(X)
    # add cluster labels to data
    data["cluster"] = kmeans.labels_
    return data


def plot_data(data):
    """
    Plot data
    """
    # get latitude and longitude
    X = data[["latitude", "longitude"]]
    # color by cluster
    plt.scatter(X["latitude"], X["longitude"], c=data["cluster"])
    plt.show()


def group_data(data):
    """
    Group data by cluster
    """
    # group by cluster
    grouped = data.groupby("cluster")
    # get summed demand per cluster
    mean_demand = grouped["avg_demand"].sum()
    # get standard deviation of demand per cluster, assume perfect correlation for worst-case variation
    std_demand = grouped["std_demand"].apply(lambda x: x.sum())
    # get number of sites per cluster
    n_sites = grouped["site_name"].count()
    # get latitude and longitude of cluster centers
    centers = grouped[["latitude", "longitude"]].mean()
    # create dataframe
    df = pd.DataFrame(
        {
            "mean_demand": mean_demand,
            "std_demand": std_demand,
            "n_sites": n_sites,
            "latitude": centers["latitude"],
            "longitude": centers["longitude"],
        }
    )
    return df


def get_mfp_data(path, n_clusters=5, plot=False):
    """
    Get data for MFP
    """
    # load data
    data = clean_data(path)
    # cluster data
    data = cluster_data(data, n_clusters)
    # plot data
    if plot:
        plot_data(data)
    # group data
    data = group_data(data)
    return data


if __name__ == "__main__":
    print(clean_data(pd.read_excel("data/mfp_food_pantries.xlsx")))
