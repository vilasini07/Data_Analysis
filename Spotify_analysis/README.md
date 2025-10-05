# 🎵 Probabilistic Modeling of Song Popularity and Mood on Spotify

## Project Overview

This project explores the underlying factors that contribute to song success and artist popularity on Spotify using bayesian modeling. The objective is to answer the following questions:
1. What are the latent correlations among features like popularity, danceability, valence, energy, and instrumentalness?
2. Which genres and mood categories dominate popular songs?
3. How can probabilistic models explain the distribution of popularity, mood, and genre?

**Approaches:** <br>
Regression: Bayesian Neural Network (BNN) to predict song popularity. <br>
Clustering: Gaussian Mixture Model (GMM) for mood-based clustering. <br>
Classification: Bayesian Logistic Regression for genre classification.

## Data Description and Analysis

**Dataset**: Spotify dataset from Kaggle (https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset). <br>
_Features_: Popularity, danceability, valence, energy, tempo, and others. <br>
**Preprocessing**: 
- Duplicates and missing values removed
- Feature scaling (StandardScaler, Min-Max, or log transformation based on distribution)
- Encoding categorical features
- Created engineered features: tempo_category and mood_bucket

**Exploratory Analysis**
1. Correlation Estimation: MCMC sampling was used to approximate posterior correlations between features. Bayesian (MCMC) heatmaps captured uncertainty, unlike Pearson’s r, which only shows linear relationships.
_Insights_:
- Danceable tracks are happier and more energetic.
- Instrumental tracks are less popular.
- Popularity correlates weakly with any single feature, suggesting multi-factor influence.
  
2. Genre Popularity: Dirichlet posterior estimation showed that dance, latin, and hip-hop dominate popular songs.
3. Popular moods: Dark Dance, Party Vibes, Balanced Energy, Moody Beats.
4. Tempo Trends: Slow-tempo songs are most popular.
5. Energy Level: Mean energy among popular tracks ≈ 0.73 (on 0–1 scale).

## Bayesian Neural Network (BNN)
**Why**: Captures uncertainty in predictions, unlike standard NNs — useful for noisy musical data. <br>
**Setup**: 13 features → 2 hidden layers (64, 32, tanh). Priors: N(0,0.5). Likelihood: Normal(mean=output, σ∼HalfNormal(5)). Inference via ADVI (PyMC). <br>
**Results**: Train 21k / Test 9k. MSE = 0.16, Uncertainty = 2.57 (std). <br>
**Notes**: Handles uncertainty and overfitting well, but ADVI is approximate and slower.

## Gaussian Mixture Model (GMM)
**Why**: Danceability and valence overlap — GMM handles soft, non-spherical clusters. <br>
**Setup**: 9 clusters, full covariance, EM algorithm (scikit-learn). <br>
**Results**: Silhouette = 0.33, ARI = 0.48. <br>
**Notes**: Flexible and probabilistic, but assumes Gaussian structure and is sensitive to initialization.

## Bayesian Logistic Regression (BLR)
**Why**: Adds uncertainty and regularization to logistic regression. <br>
**Setup**: 11 features, priors N(0,1), ADVI (50k iters), PyMC. 10k samples (70/30 split). <br>
**Results**: Accuracy = 21%. <br>
**Notes**: Limited by linearity and overlapping genres; ADVI may underestimate uncertainty.

## Conclusion
Probabilistic models reveal how musical features, mood, and genre link to popularity — offering uncertainty-aware insights for real-world music analytics.
