import numpy as np
from data_gen import *
from scipy.cluster.hierarchy import  dendrogram
from sklearn.cluster import AgglomerativeClustering
from utils import draw_disturbution
import matplotlib.pyplot as plt
train_data=read_data()
train_data,data_min,data_max=normalize(train_data)
aggcluster= AgglomerativeClustering(distance_threshold=0,n_clusters=None,linkage='ward')
aggcluster.fit(train_data)
print(aggcluster.labels_)
print(aggcluster.n_clusters)
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

plot_dendrogram(aggcluster,truncate_mode='level',p=5)
plt.show()
aggcluster= AgglomerativeClustering(n_clusters=2,linkage='ward')
aggcluster.fit(train_data)
draw_disturbution(unnormalize(data=train_data,min=data_min,max=data_max),aggcluster.labels_)
