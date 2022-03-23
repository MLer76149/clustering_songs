import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle


# plotting continuos variables
def plot_continous(df):
    for item in df.columns:
        sns.displot(x=item, data = df, kde=True)
    plt.show()
    
# plotting discrete variables    
def plot_discrete(df):
    r = math.ceil(df.shape[1]/2)
    c = 2
    fig, ax = plt.subplots(r,c, figsize=(15,40))
    i = 0
    j = 0
    for item in df.columns:
        sns.histplot(x=item, data = df, ax = ax[i, j])
        if j == 0:
            j = 1
        elif j == 1:
            j = 0
            i = i + 1
    plt.show()
    
# plotting boxplot    
def boxplot_continous(df):
    r = math.ceil(df.shape[1]/2)
    c = 2
    fig, ax = plt.subplots(r,c, figsize=(15,20))
    i = 0
    j = 0

    for item in df.columns:
        sns.boxplot(x=item, data=df, ax=ax[i, j])
        if j == 0:
            j = 1
        elif j == 1:
            j = 0
            i = i + 1
    plt.show()    
    
# transform variables and plot the transformed variables
def plot_transformer(df): 
    data_log = pd.DataFrame()
    data_log = log_it(df)
    data_bc, data_yj = power_transform(df)
    r = df.shape[1]
    c = 4
    fig, ax = plt.subplots(r, c, figsize=(30,30))
    i = 0
    data = ""
    for item in df.columns:
        for j in range(c):
            if j == 0:
                data = df
                head = "original"
            elif j == 1:
                data = data_log
                head = "log"
            elif j == 2:
                data = data_yj
                head = "yeo-johnson"
            elif j == 3:
                data = data_bc
                head = "box-cox"
            ax[0,j].set_title(head)
         
            if item in data.columns:
                sns.distplot(a = data[item], ax = ax[i, j]) 
        i = i + 1
    plt.show()
    
# perform log transform        
def log_it(df):
    data_log = pd.DataFrame()
    for item in df.columns:
        data_log[item] = df[item].apply(__log_transform_clean)
    return data_log

def __log_transform_clean(x):
    if np.isfinite(x) and x!=0:
        return np.log(x)
    else:
        return np.NAN
    
def __df_box_cox(df):
    df1 = pd.DataFrame()
    for item in df.columns:
        if df[item].min() > 0:
            df1[item] = df[item]
    return df1

def standard_scaler(df, filename, fit = True):
    df_num = df.drop(columns=["songname", "artist", "album", "id", "uri", "track_href"])
    if fit:
        scaler = StandardScaler()
        scaler.fit(df_num)
        filename = filename + ".sav"
        pickle.dump(scaler, open("scaler/"+filename, 'wb'))
        scaled = scaler.transform(df_num)
        scaled_df = pd.DataFrame(scaled, columns=df_num.columns)
        return scaled_df, filename
    if fit == False:
        loaded_model = pickle.load(open("scaler/"+filename, 'rb'))
        scaled = loaded_model.transform(df_num)
        scaled_df = pd.DataFrame(scaled, columns=df_num.columns)
        return scaled_df
    
def clustering(df):
    K = range(2, 21)
    inertia = []
    silhouette = []

    for k in K:
        print("Training a K-Means model with {} neighbours! ".format(k))
        print()
        kmeans = KMeans(n_clusters=k,
                        random_state=1234)
        kmeans.fit(df)
        filename = "models/kmeans_" + str(k) + ".sav"
        with open(filename, "wb") as f:
            pickle.dump(kmeans,f)
        inertia.append(kmeans.inertia_)
        silhouette.append(silhouette_score(df, kmeans.predict(df)))


    fig, ax = plt.subplots(1,2,figsize=(16,8))
    ax[0].plot(K, inertia, 'bx-')
    ax[0].set_xlabel('k')
    ax[0].set_ylabel('inertia')
    ax[0].set_xticks(np.arange(min(K), max(K)+1, 1.0))
    ax[0].set_title('Elbow Method showing the optimal k')
    ax[1].plot(K, silhouette, 'bx-')
    ax[1].set_xlabel('k')
    ax[1].set_ylabel('silhouette score')
    ax[1].set_xticks(np.arange(min(K), max(K)+1, 1.0))
    ax[1].set_title('Silhouette Method showing the optimal k')

def predict(df, filename, k_number):
    loaded_model = pickle.load(open("models/"+filename, 'rb'))
    cluster = loaded_model.predict(df)
    column = "cluster_k_"+str(k_number)
    df[column] = cluster
    return df

def clustering_2(df):
    K = range(2, 21)
    inertia5 = []
    inertia10 = []
    inertia30 = []
    inertia50 = []
    silhouette5 = []
    silhouette10 = []
    silhouette30 = []
    silhouette50 = []
    init = [5, 10, 30, 50]
 
    for k in K:
        for n in init:
            print("Training a K-Means model with {} neighbours and {} n! ".format(k,n))
            print()
            kmeans = KMeans(n_clusters=k, n_init=n, random_state=1234)
            kmeans.fit(df)
            filename = "models/kmeans_" + str(k) + "_" + str(n) + ".sav"
            with open(filename, "wb") as f:
                pickle.dump(kmeans,f)
            if n == 5:
                inertia5.append(kmeans.inertia_)
                silhouette5.append(silhouette_score(df, kmeans.predict(df)))
            elif n == 10:
                inertia10.append(kmeans.inertia_)
                silhouette10.append(silhouette_score(df, kmeans.predict(df)))
            elif n == 30:
                inertia30.append(kmeans.inertia_)
                silhouette30.append(silhouette_score(df, kmeans.predict(df)))
            elif n == 50:
                inertia50.append(kmeans.inertia_)
                silhouette50.append(silhouette_score(df, kmeans.predict(df)))
                
            

    fig, ax = plt.subplots(1,2,figsize=(16,8))
    ax[0].plot(K, inertia5, 'bx-', K, inertia10, 'gx-', K, inertia30, 'rx-', K, inertia50, 'yx-')
    ax[0].set_xlabel('k')
    ax[0].set_ylabel('inertia')
    ax[0].set_xticks(np.arange(min(K), max(K)+1, 1.0))
    ax[0].set_title('Elbow Method showing the optimal k, blue:5, green: 10, red: 30, yellow: 50')
    ax[1].plot(K, silouhette5, 'bx-', K, silouhette10, 'gx-', K, silouhette30, 'rx-', K, silouhette50, 'yx-')
    ax[1].set_xlabel('k')
    ax[1].set_ylabel('silhouette score')
    ax[1].set_xticks(np.arange(min(K), max(K)+1, 1.0))
    ax[1].set_title('Silhouette Method showing the optimal k, , blue:5, green: 10, red: 30, yellow: 50')
    


