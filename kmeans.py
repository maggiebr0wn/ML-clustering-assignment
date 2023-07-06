from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
# Load image
import imageio


# Set random seed so output is all same
np.random.seed(1)

def pairwise_dist(x, y):
    """
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
        dist: N x M array, where dist2[i, j] is the euclidean distance between 
        x[i, :] and y[j, :]
    """
    xi = x[:, np.newaxis, :]
    dist = np.linalg.norm(xi - y, axis = -1)

    return(dist)

    raise NotImplementedError

class KMeans(object):
    
    def __init__(self): #No need to implement
        pass

    def _init_centers(self, points, K, **kwargs): # [5 pts]
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            kwargs: any additional arguments you want
        Return:
            centers: K x D numpy array, the centers. 
        """
        k = np.minimum(K, (np.unique(points)).size)

        N = points.shape[0] # points
        D = points.shape[1] # dim

        maxp = np.max(points, axis=0)

        centers = maxp * np.random.rand(k, D)

        return centers

        raise NotImplementedError

    def _update_assignment(self, centers, points): # [10 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            points: NxD numpy array, the observations
        Return:
            cluster_idx: numpy array of length N, the cluster assignment for each point
            
        Hint: You could call pairwise_dist() function.
        """
        dist = pairwise_dist(points, centers)
        cluster_idx = np.argmin(dist, axis=1)

        return cluster_idx

        raise NotImplementedError

    def _update_centers(self, old_centers, cluster_idx, points): # [10 pts]
        """
        Args:
            old_centers: old centers KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            centers: new centers, K x D numpy array, where K is the number of clusters, and D is the dimension.
        """

        K = old_centers.shape[0] # clusters
        D = old_centers.shape[1] # dims

        centers = np.empty((K, D))

        center_list = []

        for i in range(0,K):
            # print(i)
            clust = points[np.argwhere(cluster_idx == i)] 
            clust_shape = clust.shape[0]
            if clust_shape > 0: 
                centers[i] = np.sum(clust, axis=0) / clust_shape # average
                center_list.append(centers[i])

        centers = np.array(center_list)

        # print(centers)

        return centers

        raise NotImplementedError

    def _get_loss(self, centers, cluster_idx, points): # [5 pts]
        """
        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            loss: a single float number, which is the objective function of KMeans. 
        """

        loss = 0

        i = 0
        for center in centers:

            cidx_ind = np.where(cluster_idx == i) # get cluster idx indices for cluster i
            cidx_points = np.take(points, cidx_ind, axis = 0) # get points in clusters assoc with center
            new_loss = np.linalg.norm(cidx_points - center)**2 # calc distance
            loss += new_loss
            i+=1
            
        return loss
        
        raise NotImplementedError
        
        
    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=False, **kwargs):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            verbose: boolean to set whether method should print loss (Hint: helpful for debugging)
            kwargs: any additional arguments you want
        Return:
            cluster assignments: Nx1 int numpy array
            cluster centers: K x D numpy array, the centers
            loss: final loss value of the objective function of KMeans
        """
        centers = self._init_centers(points, K, **kwargs)
        for it in range(max_iters):
            cluster_idx = self._update_assignment(centers, points)
            centers = self._update_centers(centers, cluster_idx, points)
            loss = self._get_loss(centers, cluster_idx, points)
            K = centers.shape[0]
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if verbose:
                print('iter %d, loss: %.4f' % (it, loss))

        return cluster_idx, centers, loss
    
def find_optimal_num_clusters(data, max_K=19): # [10 pts]
    np.random.seed(1)
    """Plots loss values for different number of clusters in K-Means

    Args:
        image: input image of shape(H, W, 3)
        max_K: number of clusters
    Return:
        None (plot loss values against number of clusters)
    """

    loss_list = []

    for i in range(1, (max_K + 1)):
        # print(i)
        loss = KMeans()(data, i)[2]
        loss_list.append(loss)

    x = range(1, (max_K + 1))
    y = loss_list

    return plt.plot(x, y)

    raise NotImplementedError


def intra_cluster_dist(cluster_idx, data, labels): # [4 pts]
    """
    Calculates the average distance from a point to other points within the same cluster
    
    Args:
        cluster_idx: the cluster index (label) for which we want to find the intra cluster distance
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        intra_dist_cluster: 1D array where the i_th entry denotes the average distance from point i 
                            in cluster denoted by cluster_idx to other points within the same cluster
    """

    cidx_points = [] 

    for i in range(0,len(labels)):
        if labels[i] == cluster_idx:
            cidx_points.append(data[i])

    intra_dist_cluster_list = []

    for j in range(0, len(cidx_points)): # current point
        dist_list = []
        for m in range(0, len(cidx_points)):
            if m != j:
                dist = np.linalg.norm(cidx_points[j] - cidx_points[m])
                dist_list.append(dist)

        intra_dist_cluster_list.append(np.average(dist_list))
    
    intra_dist_cluster = np.array(intra_dist_cluster_list)

    return intra_dist_cluster

    raise NotImplementedError

def inter_cluster_dist(cluster_idx, data, labels): # [4 pts]
    """
    Calculates the average distance from one cluster to the nearest cluster
    Args:
        cluster_idx: the cluster index (label) for which we want to find the intra cluster distance
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        inter_dist_cluster: 1D array where the i-th entry denotes the average distance from point i in cluster
                            denoted by cluster_idx to the nearest neighboring cluster
    """

    #print("data", data)



    clust_dict = {} # {clust : [points]}

    for i in range(0, len(labels)):
        clust = labels[i]
        points = list(data[i])
        if clust not in clust_dict.keys():
            clust_list = []
            clust_list.append(points)
            clust_dict[clust] = clust_list
        elif clust in clust_dict.keys():
            clust_list = clust_dict[clust]
            clust_list.append(points)
            clust_dict[clust] = clust_list

    # Find nearest neighboring cluster using pairwise_dist?

    # 1.0 average values per cluster
    clust_avgs_dict = {} # {clust: [avg, avg, avg]}
    for clust in clust_dict.keys():
        clust_arr = np.array(clust_dict[clust])
        clust_avg = np.average(clust_arr, axis = 0)
        clust_avgs_dict[clust] = clust_avg

    # 2.0 find closest cluster for cidx

    cidx_pts = clust_dict[cluster_idx]

    clust_dist_dict = {} # {(cidx - clust): []}

    for clust in clust_avgs_dict.keys():
        if clust != cluster_idx:
            dist = np.sqrt(np.sum((np.array(clust_avgs_dict[cluster_idx])-np.array(clust_avgs_dict[clust]))**2))  
            clust_dist_dict[clust] = dist

    closest_cluster = min(clust_dist_dict.items(), key=lambda x: x[1])[0]

    # 3.0 compute dist between points in cidx to nearest cluster

    inter_cluster_dist_list = []

    for cidx_points in clust_dict[cluster_idx]:
        dists = []
        for point_set in clust_dict[closest_cluster]:
            dist = np.linalg.norm(np.array(cidx_points) - np.array(point_set), axis = -1)

            dists.append(dist)

        avg_dists = sum(dists) / len(dists)
        inter_cluster_dist_list.append(avg_dists)        

        #print(cidx_points)

        #print(clust_dict[closest_cluster])

        #dist = average(np.linalg.norm(np.array(cidx_points) - np.array(clust_dict[closest_cluster])))

        #inter_cluster_dist_list.append(dist)

    #print(inter_cluster_dist_list)

    inter_dist_cluster = np.array(inter_cluster_dist_list)

    #print(inter_dist_cluster)

    """
    inter_cluster_dist_list = []
   
    for cidx_points in clust_dict[cluster_idx]:
        dist_list = []
        for n in clust_dict.keys():
            if n != cluster_idx:
                dist = np.linalg.norm(np.array(cidx_points) - np.array(clust_dict[n]))
                dist_list.append(dist)
        min_dist = min(dist_list)
        inter_cluster_dist_list.append(min_dist)

    inter_dist_cluster = np.array(inter_cluster_dist_list)
    """

    return inter_dist_cluster

    raise NotImplementedError

def normalized_cut(data, labels): #[2 pts]
    """
    Finds the normalized_cut of the current cluster assignment
    
    Args:
        data: NxD numpy array, where N is # points and D is the dimensionality
        labels: 1D array of length N where each number indicates of cluster assignement for that point
    Return:
        normalized_cut: normalized cut of the current cluster assignment
    """

    normalized_cut = 0

    for i in range(0, len(set(labels))):
        intra = intra_cluster_dist(i, data, labels)
        inter = inter_cluster_dist(i, data, labels)

        normalized_cut_cluster = sum(np.divide(inter, np.add(intra, inter)))
        normalized_cut = np.add(normalized_cut_cluster, normalized_cut)

    return normalized_cut

    raise NotImplementedError









