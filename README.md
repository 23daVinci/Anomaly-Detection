# Anomaly-Detection

🧱 Phase 1: Fundamentals of Anomaly Detection
1. Understanding Anomaly Detection
Types of anomalies: point, contextual, collective

Supervised vs. unsupervised vs. semi-supervised anomaly detection

Common use cases and challenges (e.g., imbalanced data)

2. Probability & Statistics Refresher
Descriptive statistics: mean, variance, skewness, kurtosis

Probability distributions: Gaussian, Poisson, Exponential, etc.

Hypothesis testing, confidence intervals

Central Limit Theorem

3. Distance and Density-Based Methods
k-Nearest Neighbors (k-NN) for anomaly detection

LOF (Local Outlier Factor)

DBSCAN for identifying clusters and noise

🔍 Phase 2: Classical ML-Based Anomaly Detection
4. Model-Based Techniques
One-Class SVM

Isolation Forest

Autoencoder-based anomaly detection (classic & denoising)

5. Time Series Anomaly Detection
Moving average, STL decomposition

ARIMA/Seasonal ARIMA (SARIMA)

Exponential Smoothing (ETS)

Change point detection (e.g., PELT, Bayesian Online Change Point Detection)

6. Statistical Process Control
Control charts (Shewhart, CUSUM, EWMA)

Process capability indices (Cp, Cpk)

🤖 Phase 3: Deep Learning for Anomaly Detection
7. Neural Networks & Representational Learning
Deep Autoencoders

Variational Autoencoders (VAE)

Generative Adversarial Networks (GANs) for anomalies

8. Time Series Deep Models
LSTM/GRU-based Autoencoders

Temporal Convolutional Networks (TCN)

Transformer-based models (Anomaly Transformer, Informer)

🧠 Phase 4: Advanced Topics
9. Unsupervised & Self-Supervised Learning
Contrastive learning (e.g., SimCLR-style representations)

Deep SVDD (Support Vector Data Description)

Self-supervised anomaly detection for time series and tabular data

10. Graph-based Anomaly Detection
Graph Neural Networks (GNNs)

Subgraph or edge-based anomaly detection

11. Multivariate Anomaly Detection
Correlation-based methods

Dimensionality reduction (PCA, t-SNE, UMAP) + anomaly scoring

Ensemble techniques (feature bagging, model bagging)

🛠️ Phase 5: Practical Implementation & Production Readiness
12. Evaluation Techniques
Precision, Recall, F1, AUC for anomaly detection

Use of Precision@K, NDCG in real-world deployment

Dealing with ground truth scarcity

13. Streaming and Real-Time Detection
Online learning algorithms (e.g., river library)

Kafka + Spark Streaming/Flink + anomaly detection pipelines

14. Explainability & Interpretability
SHAP, LIME, and interpretability for unsupervised models

Root cause analysis tools and techniques

15. Deployment & Monitoring
Building scalable detection pipelines (batch/streaming)

Alert fatigue reduction (e.g., alert prioritization, correlation)

Model monitoring and retraining strategies

📚 Optional Topics (Based on Interests & Domain)
Anomaly detection in cybersecurity (SIEM, EDR, logs)

Financial fraud detection

Industrial/IoT applications

NLP-based anomaly detection (e.g., rare sequences or intents)

Multimodal anomaly detection (e.g., image + metadata)
