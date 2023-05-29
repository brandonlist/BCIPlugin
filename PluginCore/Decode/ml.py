import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import manifold
import torch
import matplotlib
def euclidean(v1,v2):
    return np.linalg.norm(v1-v2)

#y should be 0,1 for now
def classifibility(features,y,d=None,dist_fn=euclidean, verbose=False):
    """

    Reference:

    example:
    cov = np.eye(2)
    dot_num = 100

    f_cls_0 = np.random.multivariate_normal(np.array([7,1]),cov,dot_num)
    y_0 = np.zeros(len(f_cls_0))
    f_cls_1 = np.random.multivariate_normal(np.array([2,1]),cov,dot_num)
    y_1 = np.ones(len(f_cls_1))

    plt.scatter(f_cls_0[:,0],f_cls_0[:,1],c='r')
    plt.scatter(f_cls_1[:,0],f_cls_1[:,1],c='b')

    feature = np.vstack([f_cls_0,f_cls_1])
    y = np.hstack([y_0,y_1])
    y = y.astype(np.int)
    score_01 = classifibility(feature,y,1)

    :param features:
    :param y:
    :param d:
    :param dist_fn:
    :param verbose:
    :return:
    """
    if type(y) is torch.Tensor:
        y = y.numpy()

    n_class = len(np.unique(y))
    class_code = dict(zip(np.unique(y),range(len(np.unique(y)))))
    transformed_y = np.zeros_like(y)
    for i,_y in enumerate(y):
        transformed_y[i] = class_code[_y]

    W = np.zeros((n_class,n_class))
    distance_M = np.zeros((len(features),len(features)))
    for i_f,feature_i in enumerate(features):
        for j_f,feature_j in enumerate(features):
            distance_M[i_f,j_f] = dist_fn(feature_i,feature_j)
    if d is None:
        d = np.mean(distance_M,axis=(0,1))
        if verbose:
            print('using minimum distance: ',d)
    for i,dists_i in enumerate(distance_M):
        for j,dists_i_j in enumerate(dists_i):
            if j==i:
                continue
            if dists_i_j<d:
                W[transformed_y[i]][transformed_y[j]] += 1
    W = W/np.linalg.norm(W)
    diag = np.sum(W.diagonal())
    return diag/(np.sum(W) - diag)

def get_Sw(features,y):
    dim_feature = features.shape[1]
    n_sample = features.shape[0]
    classes = np.unique(y)
    n_classes = len(classes)

    means = np.zeros((n_classes,dim_feature))

    for i_cls in range(n_classes):
        means[i_cls] = np.mean(features[y==classes[i_cls]],axis=0)

    Sws = np.zeros((len(classes),dim_feature,dim_feature))
    Sws_norm = np.zeros((len(classes),dim_feature,dim_feature))

    for i_cls,cls in enumerate(classes):
        ft = features[y==cls]
        ft -= np.mean(ft,axis=0)
        Sw = np.matmul(ft.transpose(),ft)
        Sws[i_cls] = Sw
        Sw /= len(ft)
        Sws_norm[i_cls] = Sw

    Sw = np.sum(Sws,axis=0)
    return Sw

def get_Sb(features,y):
    dim_feature = features.shape[1]
    n_sample = features.shape[0]
    classes = np.unique(y)
    n_classes = len(classes)

    Ps = np.array([np.sum(y==cls)/n_sample for cls in classes])
    means = np.zeros((n_classes, dim_feature))

    for i_cls,cls in enumerate(classes):
        means[i_cls] = np.mean(features[y == classes[i_cls]], axis=0)

    overall_mean = np.mean(features,axis=0)

    Sb = np.zeros((dim_feature, dim_feature))
    for i_cls,cls in enumerate(classes):
        sub = means[i_cls] - overall_mean
        sub = np.expand_dims(sub, axis=1)
        Sb += Ps[i_cls] * np.matmul(sub, sub.transpose())
    return Sb

def get_Jd(features,y):
    """

    higher value means more discriminant feature
    :param features:
    :param y:
    :return:
    """
    Sw = get_Sw(features,y)
    Sb = get_Sb(features,y)
    return np.trace(Sw + Sb)

def visualize_2d_feature(feature_2d,idxs,legends,colors=None):
    plt.figure()
    s_plots = []
    for i,idx in enumerate(idxs):
        if colors is None:
            s = plt.scatter(feature_2d[idx, 0], feature_2d[idx, 1])
        else:
            norm = matplotlib.colors.Normalize(vmin=colors.min(), vmax=colors.max())
            sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
            sm.set_array([])
            plt.colorbar(sm, label='normalized_time', orientation='horizontal')
            s = plt.scatter(feature_2d[idx, 0], feature_2d[idx, 1], c=colors, cmap='plasma')
        s_plots.append(s)
    if legends is not None:
        plt.legend(s_plots, legends, loc='best')

def pca_visualize(feature,idxs,legends=None):
    pca = PCA(n_components=2)
    feature_2d = pca.fit_transform(feature)
    visualize_2d_feature(feature_2d=feature_2d,idxs=idxs,legends=legends)

def tsne_visualize(feature,idxs,legends=None,colors=None):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    low = tsne.fit_transform(feature)

    low_norm = (low - low.min()) / (low.max() - low.min())
    visualize_2d_feature(feature_2d=low_norm,idxs=idxs,legends=legends,colors=colors)

def visualize_cls(feature,y,vis_func,legends=None):
    classes = np.unique(y)
    idxs = [y==i for i in classes]
    vis_func(feature=feature,idxs=idxs,legends=legends)

def get_hist(feature, min_v=None, max_v=None):
    if max_v is None:
        max_v = feature.max()
    if min_v is None:
        min_v = feature.min()

    hist, bins = np.histogram(feature, range=(min_v, max_v), density=True)
    return hist, bins

def plot_distribution(feature, min_v=None, max_v=None):
    hist, bins = get_hist(feature,min_v,max_v)
    mean_bins = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        mean_bins[i] = (bins[i] + bins[i+1]) / 2
    plt.plot(mean_bins,hist)
