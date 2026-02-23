flowchart TD
    ML([🤖 Machine Learning]) --> TRAD[Traditional ML]
    ML --> DL([Deep Learning - see Diagram 2])

    TRAD --> SUP[Supervised]
    TRAD --> UNSUP[Unsupervised]
    TRAD --> RL[Reinforcement Learning]

    SUP --> REG[Regression]
    SUP --> CLF[Classification]
    SUP --> ENS[Ensemble Methods]

    REG --> R1[Linear Regression]
    REG --> R2[Ridge / Lasso / ElasticNet]
    REG --> R3[SVR]
    REG --> R4[Polynomial Regression]

    CLF --> C1[Logistic Regression]
    CLF --> C2[SVM]
    CLF --> C3[KNN]
    CLF --> C4[Naive Bayes]
    CLF --> C5[Decision Tree]

    ENS --> E1[XGBoost]
    ENS --> E2[LightGBM]
    ENS --> E3[CatBoost]
    ENS --> E4[Random Forest]
    ENS --> E5[AdaBoost]
    ENS --> E6[Stacking / Blending]

    UNSUP --> CLU[Clustering]
    UNSUP --> DIM[Dim Reduction]
    UNSUP --> ANO[Anomaly Detection]

    CLU --> K1[K-Means]
    CLU --> K2[DBSCAN / HDBSCAN]
    CLU --> K3[Hierarchical]
    CLU --> K4[GMM]

    DIM --> D1[PCA]
    DIM --> D2[t-SNE]
    DIM --> D3[UMAP]
    DIM --> D4[ICA / NMF]

    ANO --> A1[Isolation Forest]
    ANO --> A2[One-Class SVM]
    ANO --> A3[LOF]
    ANO --> A4[Autoencoder-based]

    RL --> RLD[Discrete Actions]
    RL --> RLC[Continuous Actions]
    RL --> RLM[Model-Based]

    RLD --> RL1[Q-Learning]
    RLD --> RL2[DQN / Double DQN]
    RLD --> RL3[Rainbow]

    RLC --> RL4[PPO]
    RLC --> RL5[SAC]
    RLC --> RL6[DDPG / TD3]
    RLC --> RL7[A3C / A2C]

    RLM --> RL8[AlphaZero / MuZero]
