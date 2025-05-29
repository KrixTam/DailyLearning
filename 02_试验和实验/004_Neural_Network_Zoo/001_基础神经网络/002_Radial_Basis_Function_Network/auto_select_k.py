# 自动选择聚类中心数
import torch
from sklearn.cluster import KMeans
from rbf_data import train_dataset
from sklearn.metrics import silhouette_score


def auto_select_k(X):
    n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100]
    sil_score_max = -1
    sse_max = -1
    best_n_clusters_sse = -1
    best_n_clusters_sil_score = -1
    for num_clusters in n_clusters:
        kmeans = KMeans(num_clusters).fit(X)
        sse = kmeans.inertia_
        sil_score = silhouette_score(X, kmeans.labels_)
        print("The SSE for %i clusters is %0.2f" % (num_clusters, sse))
        print("The average silhouette score for %i clusters is %0.2f" % (num_clusters, sil_score))
        if sse > sse_max:
            sse_max = sse
            best_n_clusters_sse = num_clusters
        if sil_score > sil_score_max:
            sil_score_max = sil_score
            best_n_clusters_sil_score = num_clusters
    return {'silhouette_k': best_n_clusters_sil_score, 'elbow_k': best_n_clusters_sse}

# def auto_select_k(X):
#     n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100]
#     silhouette_avg = []
#     sil_score_max = -1
#     best_n_clusters = -1
#     for num_clusters in n_clusters:
#         kmeans = KMeans(num_clusters)
#         # kmeans = KMeans(n_clusters=num_clusters, init="k-means++", max_iter=100, n_init=1)
#         labels = kmeans.fit_predict(X)
#         sil_score = silhouette_score(X, labels)
#         print("The average silhouette score for %i clusters is %0.2f" % (num_clusters, sil_score))
#         silhouette_avg.append(sil_score)
#         if sil_score > sil_score_max:
#             sil_score_max = sil_score
#             best_n_clusters = num_clusters
#     return best_n_clusters


# def auto_select_k(X):
#     n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100]
#     sil_score_max = -1
#     sse_max = -1
#     best_n_clusters_sse = -1
#     best_n_clusters_sil_score = -1
#     for num_clusters in n_clusters:
#         kmeans = KMeans(num_clusters)
#         labels = kmeans.fit_predict(X)
#         sil_score = silhouette_score(X, labels)
#         sse = kmeans.fit(X).inertia_
#         print("The SSE for %i clusters is %0.2f" % (num_clusters, sse))
#         print("The average silhouette score for %i clusters is %0.2f" % (num_clusters, sil_score))
#         if sse > sse_max:
#             sse_max = sse
#             best_n_clusters_sse = num_clusters
#         if sil_score > sil_score_max:
#             sil_score_max = sil_score
#             best_n_clusters_sil_score = num_clusters
#     return {'silhouette_k': best_n_clusters_sil_score, 'elbow_k': best_n_clusters_sse}

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = train_dataset.data.reshape(-1, 784).double().to(device) / 255.0
    X = X_train.cpu().numpy()
    res = auto_select_k(X)
    print(res)
