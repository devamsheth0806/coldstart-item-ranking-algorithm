# -*- coding: utf-8 -*-
"""
@author: devam
"""

# PREPARING ENVIRONMENT    
from __future__ import division
import pandas as pd
import math
import numpy as np
import collections
import warnings
from surprise import SVD, Reader, accuracy, Dataset
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from collections import defaultdict
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# LOADING DATASET    
data_pd = pd.read_csv("useritemmatrix.csv")
data_pd.rename(columns={'userId': 'user_id', 'itemId': 'item_id', 'interaction': 'raw_ratings'}, inplace=True)
data_pd = data_pd.sample(n=50000, random_state=123)
data = Dataset.load_from_df(data_pd[['user_id', 'item_id', 'raw_ratings']], reader=Reader(rating_scale=(0, 1)))

print('Loaded dataset')

# HYPERPARAMETERS
factors = 100
reg_b = 1e-08
reg_q = 1e-05

# SELECTING COLD USERS
print('selecting cold users')
user_freq_df = pd.DataFrame.from_dict(collections.Counter(data_pd['user_id']), orient='index').reset_index()
user_freq_df = user_freq_df.rename(columns={'index': 'user_id', 0: 'freq'})
nr_of_cold_users = int(math.floor(len(user_freq_df) * 0.1))
cold_users = user_freq_df.sort_values(by='freq', ascending=False).head(nr_of_cold_users)['user_id']

# CANDIDATE ITEMS
item_freq_df = data_pd['item_id'].value_counts().reset_index()
item_freq_df.columns = ['item_id', 'freq']
candidate_item_list = item_freq_df[item_freq_df['freq'] >= 10]['item_id'].tolist()

# SPLIT DATA
print('splitting data')
cold_users_mask = data_pd['user_id'].isin(cold_users)
data_pd_warm = data_pd[~cold_users_mask]
df_sampled = data_pd_warm.sample(frac=1, random_state=123)
split_ratio = 0.3
num_test_samples = int(len(data_pd_warm) * split_ratio)
y_change_train_data_df = df_sampled.iloc[num_test_samples:]
y_change_test_data_df = df_sampled.iloc[:num_test_samples]

# TRAIN SVD MODEL
print('training SVD model')
model = SVD(n_factors=factors, n_epochs=100, biased=True, reg_bu=reg_b, reg_bi=reg_b, reg_pu=reg_q, reg_qi=reg_q, random_state=123)
y_change_train_data = Dataset.load_from_df(y_change_train_data_df[['user_id', 'item_id', 'raw_ratings']], Reader(rating_scale=(0, 1))).build_full_trainset()
model.fit(y_change_train_data)

# PREDICT TEST RATINGS
print('predicting ratings')
y_change_test_data_df.loc[:, 'predicted_rating'] = y_change_test_data_df.apply(
    lambda row: model.predict(row['user_id'], row['item_id']).est, axis=1)

# EXTRACT ITEM EMBEDDINGS AND CLUSTER
print('extracting embedding and clustering')
item_inner_ids = model.trainset._raw2inner_id_items
item_embeddings = []
item_ids = []
for raw_iid in candidate_item_list:
    if raw_iid in item_inner_ids:
        inner_id = item_inner_ids[raw_iid]
        item_ids.append(raw_iid)
        item_embeddings.append(model.qi[inner_id])
item_embeddings = np.array(item_embeddings)

# Automatically determine optimal number of clusters using silhouette score
sil_scores = []
best_score = -1
best_n_clusters = 2
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=123)
    labels = kmeans.fit_predict(item_embeddings)
    score = silhouette_score(item_embeddings, labels)
    sil_scores.append((n_clusters, score))
    if score > best_score:
        best_score = score
        best_n_clusters = n_clusters
        
# Plot silhouette scores
clusters, scores = zip(*sil_scores)
plt.figure(figsize=(8, 5))
plt.plot(clusters, scores, marker='o')
plt.title('Silhouette Score vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.savefig("cluster_silhouette_scores.png")
plt.close()

print(f"Best number of clusters: {best_n_clusters} with silhouette score: {best_score:.4f}")

        
kmeans = KMeans(n_clusters=best_n_clusters, random_state=123)
item_clusters = kmeans.fit_predict(item_embeddings)
item_cluster_df = pd.DataFrame({'item_id': item_ids, 'cluster': item_clusters})

pca = PCA(n_components=2)
item_2d = pca.fit_transform(item_embeddings)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(item_2d[:, 0], item_2d[:, 1], c=item_clusters, cmap=plt.cm.get_cmap('tab10', best_n_clusters), s=10)
plt.title("Clustering of Item Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
cbar = plt.colorbar(scatter, ticks=range(n_clusters))
cbar.set_label('Cluster')
plt.savefig("item_clusters_pca.png")
plt.close()

# GROUP CANDIDATE ITEMS BY CLUSTER
print('group candidates by cluster')
clustered_items = item_cluster_df.groupby('cluster')['item_id'].apply(list).to_dict()
cluster_scores_ychange = defaultdict(list)
cluster_scores_cvchange = defaultdict(list)

rating_list = [0, 1]

for cluster, items in clustered_items.items():
    for item in items:
        y_scores, e_scores = [], []
        for y in rating_list:
            temp_data = pd.concat([y_change_train_data_df, pd.DataFrame({'user_id': [0], 'item_id': [item], 'raw_ratings': [y]})])
            temp_train = Dataset.load_from_df(temp_data[['user_id', 'item_id', 'raw_ratings']], Reader(rating_scale=(0, 1))).build_full_trainset()
            model.fit(temp_train)

            y_diff, e_diff = 0, 0
            for _, row in y_change_test_data_df.iterrows():
                pred = model.predict(row['user_id'], row['item_id']).est
                y_diff += (row['predicted_rating'] - pred) ** 2
                e_diff += (row['raw_ratings'] - pred) ** 2

            y_scores.append(y_diff)
            e_scores.append(e_diff)

        cluster_scores_ychange[cluster].append({
            'item_id': item,
            'risky': min(y_scores),
            'moderate': sum(y_scores)/len(y_scores),
            'conservative': max(y_scores)
        })

        cluster_scores_cvchange[cluster].append({
            'item_id': item,
            'risky': min(e_scores),
            'moderate': sum(e_scores)/len(e_scores),
            'conservative': max(e_scores)
        })
        
# EVALUATION: RECOMMENDATION FOR ALL COLD USERS
user_cluster_map = {}
user_vectors = {}

for uid in cold_users:
    if uid in model.trainset._raw2inner_id_users:
        user_inner_id = model.trainset.to_inner_uid(uid)
        user_vectors[uid] = model.pu[user_inner_id]
    else:
        user_vectors[uid] = np.random.normal(scale=0.1, size=model.pu.shape[1])

    similarities = cosine_similarity([user_vectors[uid]], item_embeddings)[0]
    most_similar_item_idx = np.argmax(similarities)
    most_similar_item_id = item_ids[most_similar_item_idx]
    user_cluster = item_cluster_df[item_cluster_df['item_id'] == most_similar_item_id]['cluster'].values[0]
    user_cluster_map[uid] = user_cluster


# EVALUATION METRICS INIT
print("Evaluating...")

optimized_alphas = pd.DataFrame(index=cold_users, columns=['risky', 'moderate', 'conservative'])
items_to_be_shown = [10, 25, 50, 100]
results_df = pd.DataFrame(columns=['risky', 'moderate', 'conservative'], index=items_to_be_shown)

# COMPUTE RESULTS
for nr in items_to_be_shown:
    for level in ['risky', 'moderate', 'conservative']:
        total_score = 0
        for uid in cold_users:
            cluster_y_df = pd.DataFrame(cluster_scores_ychange[user_cluster_map[uid]])
            cluster_cv_df = pd.DataFrame(cluster_scores_cvchange[user_cluster_map[uid]])
            hybrid_cluster_df = pd.merge(cluster_y_df[['item_id', level]], cluster_cv_df[['item_id', level]], on='item_id', suffixes=('_y', '_cv'))
            hybrid_cluster_df['Y_norm'] = (hybrid_cluster_df[f'{level}_y'] - hybrid_cluster_df[f'{level}_y'].min()) / (hybrid_cluster_df[f'{level}_y'].max() - hybrid_cluster_df[f'{level}_y'].min() + 1e-9)
            hybrid_cluster_df['CV_norm'] = (hybrid_cluster_df[f'{level}_cv'] - hybrid_cluster_df[f'{level}_cv'].min()) / (hybrid_cluster_df[f'{level}_cv'].max() - hybrid_cluster_df[f'{level}_cv'].min() + 1e-9)

            # EM OPTIMIZATION
            alpha = 0.5
            prev_score = float('inf')
            for _ in range(50):
                hybrid_cluster_df['HybridScore'] = alpha * hybrid_cluster_df['Y_norm'] + (1 - alpha) * hybrid_cluster_df['CV_norm']
                numerator = np.dot(hybrid_cluster_df['Y_norm'] - hybrid_cluster_df['CV_norm'], 1 - 2 * hybrid_cluster_df['HybridScore'])
                denominator = np.dot(hybrid_cluster_df['Y_norm'] - hybrid_cluster_df['CV_norm'], hybrid_cluster_df['Y_norm'] - hybrid_cluster_df['CV_norm'])
                if denominator == 0:
                    break
                new_alpha = min(max(numerator / denominator, 0), 1)
                score = hybrid_cluster_df['HybridScore'].mean()
                if abs(score - prev_score) < 1e-7:
                    break
                alpha = new_alpha
                prev_score = score

            best_alpha = alpha
            optimized_alphas.at[uid, level] = best_alpha
            hybrid_cluster_df['HybridScore'] = best_alpha * hybrid_cluster_df['Y_norm'] + (1 - best_alpha) * hybrid_cluster_df['CV_norm']
            hybrid_cluster_df.sort_values(by='HybridScore', inplace=True)

            top_items = hybrid_cluster_df.head(nr)
            avg_score = top_items['HybridScore'].mean()
            total_score += avg_score

        results_df.loc[nr, level] = total_score / len(cold_users)

print("Evaluation results (lower is better for hybrid score):")
print(results_df)

results_df.to_csv("evaluation_results.csv")
optimized_alphas.to_csv("optimized_alphas.csv")
