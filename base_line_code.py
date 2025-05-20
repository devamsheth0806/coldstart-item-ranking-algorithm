# -*- coding: utf-8 -*-
"""
@author: devam
"""

# PREPARING ENVIRONMENT    

from __future__ import division
import pandas as pd
import math as math
import numpy as np
import collections
import warnings
from surprise import SVD, Reader, accuracy, Dataset
    
# LOADING AND TRANSFORMING DATASET    

# Load dataset
data_pd = pd.read_csv("useritemmatrix.csv")

# Rename columns to fit surprise package
data_pd.rename(columns={'userId': 'user_id', 'itemId': 'item_id', 'interaction': 'raw_ratings'}, inplace=True)

print('Loaded dataset')

# Reduce dataset size to 100,000 interactions for easy testing
sample_size = 50000
data_pd = data_pd.sample(n=sample_size, random_state=123)

# Transform data for surprise package
data = Dataset.load_from_df(data_pd[['user_id', 'item_id', 'raw_ratings']], reader = Reader(rating_scale=(0, 1))) 

print('Prepared dataset')

# HYPERPARAMETER TUNING

# Record optimal hyperparameters
factors = 100
reg_b = 1e-08
reg_q = 1e-05

print('Performed hyperparameter tuning')
print('Optimal number of factors: ' + str(factors))
print('Optimal bias regularization strength: ' + str(reg_b))
print('Optimal latent factor regularization strength: ' + str(reg_q))

# RANDOMLY SELECTING COLD USERS

# Order users by interaction amount
user_freq_df = pd.DataFrame.from_dict(collections.Counter(data_pd['user_id']),orient='index').reset_index()
user_freq_df = user_freq_df.rename(columns={'index':'user_id', 0:'freq'})

# Define percentage of users to be set as cold users
perc_cold_users = 0.1
nr_of_cold_users = int(math.floor(len(user_freq_df)*perc_cold_users))

# Select the [nr_of_cold_users] with the highest number of interactions
cold_users = user_freq_df.sort_values(by='freq',ascending=False).head(nr_of_cold_users)
cold_users = cold_users.iloc[:, 0]

print('Selected ' + str(nr_of_cold_users) + ' cold users')


# SETTINGS FOR SHOWN ITEMS

# Compute interaction frequency per item
item_freq_counter = collections.Counter(data_pd['item_id'])
item_freq_df = pd.DataFrame.from_dict(item_freq_counter,orient='index').reset_index()
item_freq_df = item_freq_df.rename(columns={'index':'item_id', 0:'freq'})

# Produce list of items with at least 10 interactions
threshold_item = 10
    
# create list of all candidate items meeting the threshold
candidate_item_list = item_freq_df[item_freq_df['freq']>=threshold_item]['item_id']
nr_of_candidate_items = int(math.floor(len(candidate_item_list)*perc_cold_users))

# create list of possible ratings
minimum_rating = 0
maximum_rating = 1
rating_list = list(range(minimum_rating,maximum_rating+1))

print('Selected ' + str(len(candidate_item_list)) + ' candidate items')

# CREATE DATASET WITHOUT COLD USER OBSERVATIONS TO PRODUCE ITEM RANKING

# Create a boolean mask to filter cold users
cold_users_mask = data_pd['user_id'].isin(cold_users)

# Use the mask to split the dataset
data_pd_cold = data_pd[cold_users_mask]
data_pd_warm = data_pd[~cold_users_mask]

# Split the dataset into training (70%) and test (30%)
split_ratio = 0.3
df_sampled = data_pd_warm.sample(frac=1, random_state=123) 
num_test_samples = int(len(data_pd_warm) * split_ratio)
y_change_train_data_df = df_sampled.iloc[num_test_samples:]
y_change_test_data_df = df_sampled.iloc[:num_test_samples]

# CREATING A SIMPLE SVD MODEL
    
# Create the SVD model
model = SVD(n_factors=factors, 
                    n_epochs=100,
                    biased=True,
                    reg_all=None,
                    lr_bu=None,
                    lr_bi=None,
                    lr_pu=None,
                    lr_qi=None,
                    reg_bu=reg_b,
                    reg_bi=reg_b,
                    reg_pu=reg_q,
                    reg_qi=reg_q,
                    random_state=123,
                    verbose=False)

print('Created SVD model')


### Y-CHANGE
    

# transform data for surprise package
y_change_train_data = Dataset.load_from_df(y_change_train_data_df[['user_id', 'item_id', 'raw_ratings']],
                            reader = Reader(rating_scale=(0, 1))) 
y_change_train_data = y_change_train_data.build_full_trainset()
    
# Train the model on the original train set
model.fit(y_change_train_data)
    
# Predict test item ratings
y_change_test_data_df.loc[:, 'predicted_rating'] = y_change_test_data_df.apply(
    lambda row: model.predict(row['user_id'], row['item_id']).est, axis=1)

# Initialize a list to store candidate items and their associated total differences
y_change_results_risky = pd.DataFrame(columns=['item_id', 'Y-change'])
y_change_results_moderate = pd.DataFrame(columns=['item_id', 'Y-change'])
y_change_results_conservative = pd.DataFrame(columns=['item_id', 'Y-change'])

# Progress tracking
idx = 0
total_candidate_items = len(candidate_item_list)

# Disable Warnings
warnings.simplefilter("ignore", category=FutureWarning)
pd.set_option('mode.chained_assignment', None)

# Iterate through each candidate training point i_x in set of interacted items

for i_x in candidate_item_list:
    
    # Add count for progress
    idx = idx + 1
    
    # Initialize variable to track the best rating
    y_change_candidate_risky = float('inf')
    y_change_candidate_moderate = 0
    y_change_candidate_conservative = 0
    
    # Iterate through each rating 'y' in 'Y' (0 and 1)
    for y in rating_list:
        
        # Create new training data by adding the candidate item with rating 'y'  
        y_change_new_train_data_pd = pd.concat([y_change_train_data_df, pd.DataFrame({'user_id': [0], 'item_id': [i_x], 'raw_ratings': [y]})])
        y_change_new_train_data = Dataset.load_from_df(y_change_new_train_data_pd[['user_id', 'item_id', 'raw_ratings']],
                                    reader = Reader(rating_scale=(0, 1))) 
        y_change_new_train_data = y_change_new_train_data.build_full_trainset()
        
        # Train recommender system using the new training data
        model.fit(y_change_new_train_data)
        
        # Initialize total difference for this candidate training point and rating
        y_change_candidate = 0
        
        # Iterate through each item i in the test set (I_test) and predict the rating
        for test_item in y_change_test_data_df['item_id'].unique():
            
            # Select test data for the current item i
            y_change_test_data_df_item = y_change_test_data_df[y_change_test_data_df['item_id'] == 	test_item]
            
            # Predict user ratings for the test item using the new training data
            y_change_predictions = model.predict(y_change_test_data_df_item.iloc[0]['user_id'], y_change_test_data_df_item.iloc[0]['item_id']).est
            
            # Calculate the squared difference in rating estimates for this candidate and rating 'y' on the test item
            y_change_candidate += ((y_change_predictions - y_change_test_data_df_item['predicted_rating'].values[0]) ** 2)
        
        # Select best / worst for risky and conservative strategies
        if y_change_candidate < y_change_candidate_risky:
            y_change_candidate_risky = y_change_candidate
        y_change_candidate_moderate += y_change_candidate      
        if y_change_candidate > y_change_candidate_conservative:
            y_change_candidate_conservative = y_change_candidate
            
    # Multiply by the normalizing constant
    y_change_candidate_moderate = (1/2)*y_change_candidate_moderate
        
    # Append the candidate item and its associated total difference to the results list
    data_to_append_risky = pd.DataFrame([{'item_id': i_x, 'Y-change': y_change_candidate_risky}])
    y_change_results_risky = pd.concat([y_change_results_risky,data_to_append_risky], ignore_index=True)
    data_to_append_moderate = pd.DataFrame([{'item_id': i_x, 'Y-change': y_change_candidate_moderate}])
    y_change_results_moderate = pd.concat([y_change_results_moderate,data_to_append_moderate], ignore_index=True)
    data_to_append_conservative = pd.DataFrame([{'item_id': i_x, 'Y-change': y_change_candidate_conservative}])
    y_change_results_conservative = pd.concat([y_change_results_conservative, data_to_append_conservative], ignore_index=True)
    print('Y-change: appended ' + 'Item ' + str(idx) + ' out of ' + str(total_candidate_items))

# Re-enable warnings
warnings.resetwarnings()
pd.set_option('mode.chained_assignment', 'warn')

# Store results
y_change_items_risky_df = y_change_results_risky
y_change_items_risky_df.sort_values(by='Y-change',inplace=True,ascending=False)
y_change_items_moderate_df = y_change_results_moderate
y_change_items_moderate_df.sort_values(by='Y-change',inplace=True,ascending=False)
y_change_items_conservative_df = y_change_results_conservative
y_change_items_conservative_df.sort_values(by='Y-change',inplace=True,ascending=False)

print('Computed Y-change scoring')

### CV-BASED

cv_based_train_data_df = df_sampled.iloc[num_test_samples:]
cv_based_test_data_df = df_sampled.iloc[:num_test_samples]
    
# transform data for surprise package
cv_based_train_data = Dataset.load_from_df(cv_based_train_data_df[['user_id', 'item_id', 'raw_ratings']],
                            reader = Reader(rating_scale=(0, 1))) 
cv_based_train_data = cv_based_train_data.build_full_trainset()

# Initialize a list to store candidate items and their associated total differences
cv_based_results_risky = pd.DataFrame(columns=['CandidateItem', 'CV_based'])
cv_based_results_moderate = pd.DataFrame(columns=['CandidateItem', 'CV_based'])
cv_based_results_conservative = pd.DataFrame(columns=['CandidateItem', 'CV_based'])

# Progress tracking
idx = 0
total_candidate_items = len(candidate_item_list)
warnings.simplefilter("ignore", category=FutureWarning)

# Iterate through each candidate training point i_x in set of interacted items Iu

for i_x in candidate_item_list:
    
    # Add count for progress
    idx = idx + 1
    
    # Initialize variable to track the best rating
    cv_based_candidate_risky = float('inf')
    cv_based_candidate_moderate = 0
    cv_based_candidate_conservative = 0
    
    # Iterate through each rating 'y' in 'Y' (0 and 1)
    for y in rating_list:
        
        # Create new training data by adding the candidate item with rating 'y'  
        cv_based_new_train_data_pd = pd.concat([cv_based_train_data_df, pd.DataFrame({'user_id': [0], 'item_id': [i_x], 'raw_ratings': [y]})])
        cv_based_new_train_data = Dataset.load_from_df(cv_based_new_train_data_pd[['user_id', 'item_id', 'raw_ratings']],
                                    reader = Reader(rating_scale=(0, 1))) 
        cv_based_new_train_data = cv_based_new_train_data.build_full_trainset()
        
        # Train recommender system using the new training data
        model.fit(cv_based_new_train_data)
        
        # Initialize total squared difference in rating estimates for this candidate item i_x and rating y
        cv_based_candidate = 0
        
        # Iterate through each item i in the test set (I_test) and predict the rating
        for i in cv_based_test_data_df['item_id'].unique():
            
            # Select test data for the current item i
            cv_based_test_data_pd_item = cv_based_test_data_df[cv_based_test_data_df['item_id'] == 	i]
            
            # Predict user ratings for the test item using the new training data
            cv_based_predictions = model.predict(cv_based_test_data_pd_item.iloc[0]['user_id'], cv_based_test_data_pd_item.iloc[0]['item_id']).est
            
            # Calculate the squared difference in rating estimates for this candidate and rating 'y' on the test item
            cv_based_candidate += ((cv_based_predictions - cv_based_test_data_pd_item['raw_ratings'].values[0]) ** 2)
        
        # Select best / worst for risky and conservative strategies
        if cv_based_candidate < cv_based_candidate_risky:
            cv_based_candidate_risky = cv_based_candidate
        cv_based_candidate_moderate += cv_based_candidate      
        if cv_based_candidate > cv_based_candidate_conservative:
            cv_based_candidate_conservative = cv_based_candidate
            
    # Multiply by the normalizing constant
    cv_based_candidate_moderate = (1/2)*y_change_candidate_moderate
        
    # Append the candidate item and its associated total difference to the results list
    data_to_append_risky = pd.DataFrame([{'item_id': i_x, 'CV_based': cv_based_candidate_risky}])
    cv_based_results_risky = pd.concat([cv_based_results_risky, data_to_append_risky], ignore_index=True)
    data_to_append_moderate = pd.DataFrame([{'item_id': i_x, 'CV_based': cv_based_candidate_moderate}])
    cv_based_results_moderate = pd.concat([cv_based_results_moderate, data_to_append_moderate], ignore_index=True)
    data_to_append_conservative = pd.DataFrame([{'item_id': i_x, 'CV_based': cv_based_candidate_conservative}])
    cv_based_results_conservative = pd.concat([cv_based_results_conservative, data_to_append_conservative], ignore_index=True)
    print('CV-based: appended ' + 'Item ' + str(idx) + ' out of ' + str(total_candidate_items))

# Store results
cv_based_items_risky_df = cv_based_results_risky
cv_based_items_risky_df.sort_values(by='CV_based',inplace=True,ascending=False)
cv_based_items_moderate_df = cv_based_results_moderate
cv_based_items_moderate_df.sort_values(by='CV_based',inplace=True,ascending=False)
cv_based_items_conservative_df = cv_based_results_conservative
cv_based_items_conservative_df.sort_values(by='CV_based',inplace=True,ascending=False)

warnings.resetwarnings()
print('Computed CV-based scoring')

### RESULTS

# Set number of items to show to the cold user
items_to_be_shown = [10, 25, 50, 100]    

# Create dataframe for results
ranking_strategies = ['Risky Y-change strategy', 'Moderate Y-change strategy', 'Conservative Y-change strategy', 'Risky CV-based strategy', 'Moderate CV-based strategy', 'Conservative CV-based strategy']
results_df = pd.DataFrame(columns=[ranking_strategies], index=[items_to_be_shown])
results_df_dich = pd.DataFrame(columns=[ranking_strategies], index=[items_to_be_shown])

# Compute results for each strategy and amount of items under consideration
for nr_of_shown_items in items_to_be_shown:
    print('Number of items shown to the cold user(s): ' + str(nr_of_shown_items))

    ### Y-CHANGE STRATEGY ###
    # Select [nr_of_shown_items] items with largest Y-change
    y_change_items_risky = y_change_items_risky_df.head(nr_of_shown_items)['item_id']
    y_change_items_risky = np.array(y_change_items_risky)
    y_change_items_moderate = y_change_items_moderate_df.head(nr_of_shown_items)['item_id']
    y_change_items_moderate = np.array(y_change_items_moderate)
    y_change_items_conservative = y_change_items_conservative_df.head(nr_of_shown_items)['item_id']
    y_change_items_conservative = np.array(y_change_items_conservative)
    print('Computed ranking using Y-change strategy')
    
    ### CV-BASED STRATEGY ###
    # Select [nr_of_shown_items] items with smallest CV-based score 
    cv_based_items_risky = cv_based_items_risky_df.head(nr_of_shown_items)['item_id']
    cv_based_items_risky = np.array(cv_based_items_risky)
    cv_based_items_moderate = cv_based_items_moderate_df.head(nr_of_shown_items)['item_id']
    cv_based_items_moderate = np.array(cv_based_items_moderate)
    cv_based_items_conservative = cv_based_items_conservative_df.head(nr_of_shown_items)['item_id']
    cv_based_items_conservative = np.array(cv_based_items_conservative)
    print('Computed ranking using CV-based strategy')
    
    ### RISKY Y-CHANGE STRATEGY ###
    
    ranking_strategy = 'Risky Y-change strategy'
    
    # Construct train set
        # Include all user-item interactions
        # Exclude cold user-item interactions
        # Except the cold user-item interactions with items under consideration
    train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(y_change_items_risky))]
    train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                                reader = Reader(rating_scale=(0, 1)))
    train = train.build_full_trainset()
    
    # Construct test set
        # Select cold user-item interactions with items under consideration
    cold_users_interacted = np.array(data_pd[(data_pd.user_id.isin(cold_users)) & (data_pd.item_id.isin(y_change_items_risky))]['user_id'])
    test_pd = data_pd[(data_pd.user_id.isin(cold_users_interacted)) & (~data_pd.item_id.isin(y_change_items_risky))]
    
    # Compute predictions
    model.fit(train)
    test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
    pred = model.test(test)
    
    # Compute dichotomized predictions
    rounded_predictions = [round(prediction.est) for prediction in pred]
    squared_errors = [(prediction.est - prediction.r_ui) ** 2 for prediction in pred]
    mean_squared_error = sum(squared_errors) / len(pred)
    rmse_d = mean_squared_error ** 0.5
    results_df_dich.loc[nr_of_shown_items, ranking_strategy] = rmse_d
    
    # Compute RMSE
    rmse = accuracy.rmse(pred)
    
    # Update results
    results_df.loc[nr_of_shown_items, ranking_strategy] = rmse
    
    ranking_strategy = 'Moderate Y-change strategy'
    
    # Construct train set
        # Include all user-item interactions
        # Exclude cold user-item interactions
        # Except the cold user-item interactions with items under consideration
    train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(y_change_items_moderate))]
    train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                                reader = Reader(rating_scale=(0, 1)))
    train = train.build_full_trainset()
    
    # Construct test set
        # Select cold user-item interactions with items under consideration
    cold_users_interacted = np.array(data_pd[(data_pd.user_id.isin(cold_users)) & (data_pd.item_id.isin(y_change_items_moderate))]['user_id'])
    test_pd = data_pd[(data_pd.user_id.isin(cold_users_interacted)) & (~data_pd.item_id.isin(y_change_items_moderate))]
    
    # Compute predictions
    model.fit(train)
    test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
    pred = model.test(test)
    
    # Compute dichotomized predictions
    rounded_predictions = [round(prediction.est) for prediction in pred]
    squared_errors = [(prediction.est - prediction.r_ui) ** 2 for prediction in pred]
    mean_squared_error = sum(squared_errors) / len(pred)
    rmse_d = mean_squared_error ** 0.5
    results_df_dich.loc[nr_of_shown_items, ranking_strategy] = rmse_d
    
    # Compute RMSE
    rmse = accuracy.rmse(pred)
    
    # Update results
    results_df.loc[nr_of_shown_items, ranking_strategy] = rmse
    
    ranking_strategy = 'Conservative Y-change strategy'
    
    # Construct train set
        # Include all user-item interactions
        # Exclude cold user-item interactions
        # Except the cold user-item interactions with items under consideration
    train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(y_change_items_conservative))]
    train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                                reader = Reader(rating_scale=(0, 1)))
    train = train.build_full_trainset()
    
    # Construct test set
        # Select cold user-item interactions with items under consideration
    cold_users_interacted = np.array(data_pd[(data_pd.user_id.isin(cold_users)) & (data_pd.item_id.isin(y_change_items_conservative))]['user_id'])
    test_pd = data_pd[(data_pd.user_id.isin(cold_users_interacted)) & (~data_pd.item_id.isin(y_change_items_conservative))]
    
    # Compute predictions
    model.fit(train)
    test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
    pred = model.test(test)
    
    # Compute dichotomized predictions
    rounded_predictions = [round(prediction.est) for prediction in pred]
    squared_errors = [(prediction.est - prediction.r_ui) ** 2 for prediction in pred]
    mean_squared_error = sum(squared_errors) / len(pred)
    rmse_d = mean_squared_error ** 0.5
    results_df_dich.loc[nr_of_shown_items, ranking_strategy] = rmse_d
    
    # Compute RMSE
    rmse = accuracy.rmse(pred)
    
    # Update results
    results_df.loc[nr_of_shown_items, ranking_strategy] = rmse
    
    print('RMSE computed for Y-change strategy')
    
    ### CV-BASED STRATEGY ###
    
    ranking_strategy = 'Risky CV-based strategy'
    
    # Construct train set
        # Include all user-item interactions
        # Exclude cold user-item interactions
        # Except the cold user-item interactions with items under consideration
    train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(cv_based_items_risky))]
    train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                                reader = Reader(rating_scale=(0, 1)))
    train = train.build_full_trainset()
    
    # Construct test set
        # Select cold user-item interactions with items under consideration
    cold_users_interacted = np.array(data_pd[(data_pd.user_id.isin(cold_users)) & (data_pd.item_id.isin(cv_based_items_risky))]['user_id'])
    test_pd = data_pd[(data_pd.user_id.isin(cold_users_interacted)) & (~data_pd.item_id.isin(cv_based_items_risky))]
    
    # Compute predictions
    model.fit(train)
    test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
    pred = model.test(test)
    
    # Compute dichotomized predictions
    rounded_predictions = [round(prediction.est) for prediction in pred]
    squared_errors = [(prediction.est - prediction.r_ui) ** 2 for prediction in pred]
    mean_squared_error = sum(squared_errors) / len(pred)
    rmse_d = mean_squared_error ** 0.5
    results_df_dich.loc[nr_of_shown_items, ranking_strategy] = rmse_d
    
    # Compute RMSE
    rmse = accuracy.rmse(pred)
    
    # Update results
    results_df.loc[nr_of_shown_items, ranking_strategy] = rmse
    
    ranking_strategy = 'Moderate CV-based strategy'
    
    # Construct train set
        # Include all user-item interactions
        # Exclude cold user-item interactions
        # Except the cold user-item interactions with items under consideration
    train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(cv_based_items_moderate))]
    train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                                reader = Reader(rating_scale=(0, 1)))
    train = train.build_full_trainset()
    
    # Construct test set
        # Select cold user-item interactions with items under consideration
    cold_users_interacted = np.array(data_pd[(data_pd.user_id.isin(cold_users)) & (data_pd.item_id.isin(cv_based_items_moderate))]['user_id'])
    test_pd = data_pd[(data_pd.user_id.isin(cold_users_interacted)) & (~data_pd.item_id.isin(cv_based_items_moderate))]
    
    # Compute predictions
    model.fit(train)
    test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
    pred = model.test(test)
    
    # Compute dichotomized predictions
    rounded_predictions = [round(prediction.est) for prediction in pred]
    squared_errors = [(prediction.est - prediction.r_ui) ** 2 for prediction in pred]
    mean_squared_error = sum(squared_errors) / len(pred)
    rmse_d = mean_squared_error ** 0.5
    results_df_dich.loc[nr_of_shown_items, ranking_strategy] = rmse_d
    
    # Compute RMSE
    rmse = accuracy.rmse(pred)
    
    # Update results
    results_df.loc[nr_of_shown_items, ranking_strategy] = rmse
    
    ranking_strategy = 'Conservative CV-based strategy'
    
    # Construct train set
        # Include all user-item interactions
        # Exclude cold user-item interactions
        # Except the cold user-item interactions with items under consideration
    train_pd = data_pd[(~data_pd.user_id.isin(cold_users)) | (data_pd.item_id.isin(cv_based_items_conservative))]
    train = Dataset.load_from_df(train_pd[['user_id', 'item_id', 'raw_ratings']],
                                reader = Reader(rating_scale=(0, 1)))
    train = train.build_full_trainset()
    
    # Construct test set
        # Select cold user-item interactions with items under consideration
    cold_users_interacted = np.array(data_pd[(data_pd.user_id.isin(cold_users)) & (data_pd.item_id.isin(cv_based_items_conservative))]['user_id'])
    test_pd = data_pd[(data_pd.user_id.isin(cold_users_interacted)) & (~data_pd.item_id.isin(cv_based_items_conservative))]
    
    # Compute predictions
    model.fit(train)
    test = list(test_pd[['user_id', 'item_id', 'raw_ratings']].itertuples(index=False, name=None))
    pred = model.test(test)
    
    # Compute dichotomized predictions
    rounded_predictions = [round(prediction.est) for prediction in pred]
    squared_errors = [(prediction.est - prediction.r_ui) ** 2 for prediction in pred]
    mean_squared_error = sum(squared_errors) / len(pred)
    rmse_d = mean_squared_error ** 0.5
    results_df_dich.loc[nr_of_shown_items, ranking_strategy] = rmse_d
    
    # Compute RMSE
    rmse = accuracy.rmse(pred)
    
    # Update results
    results_df.loc[nr_of_shown_items, ranking_strategy] = rmse
    
    print('RMSE computed for CV-based strategy')
    print('Completed evaluation')
