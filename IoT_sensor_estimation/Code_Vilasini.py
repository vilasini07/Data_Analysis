# BAYESIAN LEARNING PROJECT
# Name: Vilasini Ashokan
# SR Number: 24034

#################################
# GMM
#################################
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# We define a list containing all the node IDs to be processed
nodes = [501, 502, 505, 507, 508]

# We define a list to store the predicted value for each node data
list_predicted=[]

# We iterate over each node to process the data and find the missing values
for node in nodes:

    # Here we read and preprocess the data for each node
    df = pd.read_csv(f'Node_{node}.csv')
    # Next the typo in year is corrected
    df['timestamp'] = df['timestamp'].str.replace('0014', '2014').str.replace('0015', '2015')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')
    # We define a new column to find corresponding unix time
    # Unix time is the number of seconds passed from January 1, 1970, 00:00:00 UTC
    # astype('int64') function finds the number of nanoseconds passed from January 1, 1970, 00:00:00 UTC and then we divide by 10**9 to convert it to seconds 
    df['unix_time'] = df['timestamp'].astype('int64') // 10**9
    # Next we'll split the data into train and test
    entries_to_predict = df[['temperature', 'humidity']].isna().any(axis=1)
    test_data = df[entries_to_predict].copy()
    test_data['missing feature'] = np.where(test_data['temperature'].isna(), 'temperature', 'humidity')
    train_data = df[~entries_to_predict]
    # We'll consider 3 features 'unix_time', 'temperature', 'humidity' as data features for GMM model
    features = ['unix_time', 'temperature', 'humidity']
    # Standardize the training data
    scaler = StandardScaler()
    scaled_train_data = scaler.fit_transform(train_data[features])
    # Next we use the training data's mean and std to scale the test data
    scaled_test_data = test_data[features].copy()
    for col in features:
        scaled_test_data[col] = (scaled_test_data[col] - scaler.mean_[features.index(col)]) / scaler.scale_[features.index(col)]

    # Fit GMM on standardized training data
    num_comp = 45
    gmm = GaussianMixture(n_components=num_comp, covariance_type='full', random_state=42)
    gmm.fit(scaled_train_data)
    # Next we define a function to predict the missing values in test data
    def predict_values(row, gmm, scaler):

        missing_entries = pd.isna(row)
        idx_of_obs_data = np.where(~missing_entries)[0]
        idx_of_missing_data = np.where(missing_entries)[0]

        # The values are predicted using the Gaussian Mixture Model as follows:
        # Suppose gamma_ik represents the responsibility of the k-th component for the i-th data point 
        # Suppose mu_k and sigma_k are the mean and covariance of the k-th component
        # Then x_missing = sum over k components (gamma_ik * E[x_missing | x_obs, k]) 
        # Now note that E[x_missing | x_obs, k] = mu_missing + sigma_cross.T @ inv(sigma_obs) @ (x_obs - mu_obs)
        resn = [] # This will store the responsilities for the missing data point with respect to each component
        exp_missing = [] # This will store the expectation of the missing data point with respect to each component
        for k in range(gmm.n_components):
            comp_k_mean = gmm.means_[k]
            comp_k_cov = gmm.covariances_[k]
            weight = gmm.weights_[k] # This is the prior probability of the k-th component
            mu_obs = comp_k_mean[idx_of_obs_data] # This is the mean of the observed features of the data point
            mu_missing = comp_k_mean[idx_of_missing_data] # This is the mean of the missing feature of the data point
            sigma_obs = comp_k_cov[np.ix_(idx_of_obs_data, idx_of_obs_data)] 
            sigma_cross = comp_k_cov[np.ix_(idx_of_obs_data, idx_of_missing_data)]
            x_obs = row.iloc[idx_of_obs_data]
            exp_missing_val = mu_missing + sigma_cross.T @ np.linalg.inv(sigma_obs) @ (x_obs - mu_obs)
            # Next we need to find the likelihood of the observed data point
            prob_x_obs = np.exp(-0.5 * (x_obs - mu_obs).T @ np.linalg.inv(sigma_obs) @ (x_obs - mu_obs)) / np.sqrt(((2 * np.pi) ** len(idx_of_obs_data)) * np.linalg.det(sigma_obs))
            # Now the responsibility i.e. p(k|x_obs) = p(x_obs|k) * p(k) / sum over j (p(x_obs|j) * p(j))
            #                                        = prob_x_obs * weight / sum over j (prob_x_obs_j * weight_j) 
            # We just find the numerator for each component in each iteration
            resn.append(weight * prob_x_obs)
            exp_missing.append(exp_missing_val)
        # Normalize responsibilities
        total_prob_x_obs = sum(resn)
        resn /= total_prob_x_obs
        predicted_val = 0
        for i in range(len(resn)):
            predicted_val += resn[i] * exp_missing[i]
        return predicted_val

    # We'll now implement the above function to predict the missing values in the test data
    for idx, row in scaled_test_data.iterrows():
        filtered_row = row[features]
        missing_entries = np.where(pd.isna(filtered_row))[0]
        pred_val = predict_values(filtered_row, gmm, scaler)
        scaled_test_data.loc[idx, features[missing_entries[0]]] = pred_val
    # here we perform inverse scaling to the original range
    pred_orig_range = scaler.inverse_transform(scaled_test_data[features])
    test_data.loc[:, features] = pred_orig_range
    # Next we merge the predicted test data back with the test_data
    df.loc[entries_to_predict, features] = test_data[features]
    original_missing = test_data[['temperature', 'humidity']].isna() 
    res = pd.DataFrame({
        'ID': test_data['ID'],
        'Prediction': test_data.apply(lambda x: x['temperature'] if x['missing feature'] == 'temperature' else x['humidity'], axis=1)
    })
    res['Prediction'] = res['Prediction'].round(4)
    list_predicted.append(res)

# Combine results from all nodes
final_result = pd.concat(list_predicted)
final_result.to_csv('submission_file.csv', index=False)