import pandas as __pd
import numpy as __np
import matplotlib.pyplot as __plt
from sklearn.preprocessing import StandardScaler as __sc
from sklearn.cluster import KMeans as __km
from sklearn.metrics import silhouette_score as __si
import pickle as __pi

def standard_scaler(df, filename, fit = True):
    df_num = df.drop(columns=["songname", "artist", "album", "id", "uri", "track_href"])
    if fit:
        scaler = __sc()
        scaler.fit(df_num)
        filename = filename + ".sav"
        __pi.dump(scaler, open("scaler/"+filename, 'wb'))
        scaled = scaler.transform(df_num)
        scaled_df = __pd.DataFrame(scaled, columns=df_num.columns)
        return scaled_df, filename
    if fit == False:
        loaded_model = __pi.load(open("scaler/"+filename, 'rb'))
        scaled = loaded_model.transform(df_num)
        scaled_df = __pd.DataFrame(scaled, columns=df_num.columns)
        return scaled_df
    
def clustering(df):
    K = range(2, 21)
    inertia = []
    silhouette = []

    for k in K:
        print("Training a K-Means model with {} neighbours! ".format(k))
        print()
        kmeans = __km(n_clusters=k,
                        random_state=1234)
        kmeans.fit(df)
        filename = "models/kmeans_" + str(k) + ".sav"
        with open(filename, "wb") as f:
            __pi.dump(kmeans,f)
        inertia.append(kmeans.inertia_)
        silhouette.append(__si(df, kmeans.predict(df)))


    fig, ax = __plt.subplots(1,2,figsize=(16,8))
    ax[0].plot(K, inertia, 'bx-')
    ax[0].set_xlabel('k')
    ax[0].set_ylabel('inertia')
    ax[0].set_xticks(__np.arange(min(K), max(K)+1, 1.0))
    ax[0].set_title('Elbow Method showing the optimal k')
    ax[1].plot(K, silhouette, 'bx-')
    ax[1].set_xlabel('k')
    ax[1].set_ylabel('silhouette score')
    ax[1].set_xticks(__np.arange(min(K), max(K)+1, 1.0))
    ax[1].set_title('Silhouette Method showing the optimal k')

def predict(df_scaled, df_original, filename):
    loaded_model = __pi.load(open("models/"+filename, 'rb'))
    cluster = loaded_model.predict(df_scaled)
    column = "cluster_"+filename[7:-4]
    df_original[column] = cluster
    return df_original

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
            kmeans = __km(n_clusters=k, n_init=n, random_state=1234)
            kmeans.fit(df)
            filename = "models/kmeans_" + str(k) + "_" + str(n) + ".sav"
            with open(filename, "wb") as f:
                __pi.dump(kmeans,f)
            if n == 5:
                inertia5.append(kmeans.inertia_)
                silhouette5.append(__si(df, kmeans.predict(df)))
            elif n == 10:
                inertia10.append(kmeans.inertia_)
                silhouette10.append(__si(df, kmeans.predict(df)))
            elif n == 30:
                inertia30.append(kmeans.inertia_)
                silhouette30.append(__si(df, kmeans.predict(df)))
            elif n == 50:
                inertia50.append(kmeans.inertia_)
                silhouette50.append(__si(df, kmeans.predict(df)))
                
            

    fig, ax = __plt.subplots(1,2,figsize=(16,8))
    ax[0].plot(K, inertia5, 'bx-', K, inertia10, 'gx-', K, inertia30, 'rx-', K, inertia50, 'yx-')
    ax[0].set_xlabel('k')
    ax[0].set_ylabel('inertia')
    ax[0].set_xticks(__np.arange(min(K), max(K)+1, 1.0))
    ax[0].set_title('Elbow Method showing the optimal k, blue:5, green: 10, red: 30, yellow: 50')
    ax[1].plot(K, silhouette5, 'bx-', K, silhouette10, 'gx-', K, silhouette30, 'rx-', K, silhouette50, 'yx-')
    ax[1].set_xlabel('k')
    ax[1].set_ylabel('silhouette score')
    ax[1].set_xticks(__np.arange(min(K), max(K)+1, 1.0))
    ax[1].set_title('Silhouette Method showing the optimal k, , blue:5, green: 10, red: 30, yellow: 50')
    


