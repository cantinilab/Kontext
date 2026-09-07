from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def gmm_clustering(n_clusters, seed, matrix, reg_covar=1e-3):
    gmm = GaussianMixture(
        n_components=n_clusters, random_state=seed, reg_covar=reg_covar
    ).fit(matrix)
    return gmm.predict(matrix)


def kmeans_clustering(n_clusters, seed, matrix):
    kmeans = KMeans(n_clusters, random_state=seed, n_init=10).fit(matrix)
    return kmeans.labels_