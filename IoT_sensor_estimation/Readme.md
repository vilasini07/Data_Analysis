# 🌡️ Probabilistic Imputation of Sensor Data

## Overview
**Dataset**: temperature and humidity from 5 sensors across 5 locations, sampled every 10s. Each sensor has 200 missing values (either temp or humidity). <br>
**Goal**: fill missing data using probabilistic models — Bayesian Regression and GMM. <br>
**Data cleaning**: Fixed incorrect year values (0014, 0015) for Unix time conversion. The data was split as: missing rows → test, others → train. Scaled features using StandardScaler.

## Model 1 – Bayesian Linear Regression

_Idea_: Simple probabilistic interpolation between features. <br>
_Approach_: Temperature & humidity modeled as linear functions of Unix time. Predictions averaged from two regressions (e.g., Temp–Time and Temp–Humidity). Regularization λ = 1. <br>
_Result_: _RMSE_: 7.74 → weak accuracy, unable to capture temporal or spatial patterns. <br>
_Pros_: Fast, simple, stable. <br>
_Cons_: Too simplistic for complex sensor data.

## Model 2 – Bayesian Polynomial Regression
_Idea_: Extend linear model to nonlinear patterns using 15th-degree polynomials. <br>
_Approach_: Temperature and humidity modeled as degree-15 polynomial functions of Unix time. Regularization λ₁ = λ₂ = 2–3. <br>
_Result_: _RMSE_: 6.53 (improvement over linear). <br>
_Pros_: Captures mild nonlinearity. <br>
_Cons_: Overfits easily, computationally heavier, limited accuracy gain.

## Model 3 – Gaussian Mixture Model
_Motivation_: Plots of temperature/humidity vs. time show clustered, non-linear relationships. GMM fits multiple Gaussian distributions — a better match for such data. <br>
_Approach_: Each point modeled as a mixture of K Gaussian components. Used EM algorithm via scikit-learn’s GaussianMixture. Missing values predicted using conditional expectations from fitted distributions. <br>
_Results_: <br>
| **Clusters (K)** | **5** | **15** | **25** | **35** | **45** |
|-------------------|-------|--------|--------|--------|--------|
| **RMSE**          | 3.63  | 2.57   | 2.21   | 2.00   | **1.92** |
<br>
Best Model: K = 45 <br>

_Why it works_: Models joint distribution of temperature, humidity, and time with multiple Gaussian components — aligning with the true clustered nature of data. <br>
_Pros_: Captures multimodal & non-linear patterns; Smooth interpolation; Handles clusters with varying shape/density <br>
_Cons_: Slower due to multiple covariance inversions; Sensitive to initialization and cluster count; Ignores spatial dependencies

## Key Learnings
- Joint probability modeling provides smoother and more realistic imputations. <br>
- Scaling and efficiency matter more than model complexity in real-world sensor data. <br>
- EM-based probabilistic models (like GMM) strike a good balance between accuracy and computation.

## Possible Improvements
- Automate cluster selection using BIC or Variational Bayesian GMM. <br>
- Incorporate spatial dependencies (e.g., Kriging) for spatiotemporal modeling. <br>
- Explore hybrid models combining GMM + temporal kernels for dynamic sensor prediction.
