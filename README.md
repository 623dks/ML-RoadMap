# ML-RoadMap
```mermaid


flowchart TD
    AI([🤖 Artificial Intelligence]) --> ML([📊 Machine Learning])

    ML --> TRAD[Traditional ML]
    ML --> DL[Deep Learning]

    TRAD --> SUP[Supervised]
    TRAD --> UNSUP[Unsupervised]
    TRAD --> RL[Reinforcement Learning]

    SUP --> REG[Regression]
    SUP --> CLF[Classification]
    SUP --> ENS[Ensemble Methods]

    REG --> REG1[Linear Regression]
    REG --> REG2[Ridge / Lasso / ElasticNet]
    REG --> REG3[SVR]
    REG --> REG4[Polynomial Regression]

    CLF --> CLF1[Logistic Regression]
    CLF --> CLF2[SVM]
    CLF --> CLF3[KNN]
    CLF --> CLF4[Naive Bayes]
    CLF --> CLF5[Decision Tree]

    ENS --> ENS1[Random Forest]
    ENS --> ENS2[XGBoost]
    ENS --> ENS3[LightGBM]
    ENS --> ENS4[CatBoost]
    ENS --> ENS5[AdaBoost]
    ENS --> ENS6[Stacking / Blending]

    UNSUP --> CLU[Clustering]
    UNSUP --> DIM[Dim Reduction]
    UNSUP --> ANO[Anomaly Detection]

    CLU --> CLU1[K-Means]
    CLU --> CLU2[DBSCAN / HDBSCAN]
    CLU --> CLU3[Hierarchical]
    CLU --> CLU4[GMM]

    DIM --> DIM1[PCA]
    DIM --> DIM2[t-SNE]
    DIM --> DIM3[UMAP]
    DIM --> DIM4[ICA / NMF]

    ANO --> ANO1[Isolation Forest]
    ANO --> ANO2[One-Class SVM]
    ANO --> ANO3[LOF]
    ANO --> ANO4[Autoencoder-based]

    RL --> RL_D[Discrete Actions]
    RL --> RL_C[Continuous Actions]
    RL --> RL_M[Model-Based]

    RL_D --> RL1[Q-Learning]
    RL_D --> RL2[DQN / Double DQN]
    RL_D --> RL3[Rainbow]

    RL_C --> RL4[PPO]
    RL_C --> RL5[SAC]
    RL_C --> RL6[DDPG / TD3]
    RL_C --> RL7[A3C / A2C]

    RL_M --> RL8[AlphaZero / MuZero]
    RL_M --> RL9[Dreamer / World Models]

    DL --> ANN[Feedforward / MLP]
    DL --> CNN_F[CNN Family]
    DL --> RNN_F[RNN Family]
    DL --> TRANS[Transformer Family]
    DL --> GEN[Generative Models]
    DL --> GNN[Graph Neural Networks]

    ANN --> ANN1[MLP / ANN]
    ANN --> ANN2[Deep Neural Network]

    CNN_F --> CNN1[LeNet → AlexNet → VGG]
    CNN_F --> CNN2[ResNet / DenseNet]
    CNN_F --> CNN3[EfficientNet / MobileNet]
    CNN_F --> CNN4[ConvNeXt]

    RNN_F --> RNN1[Vanilla RNN]
    RNN_F --> RNN2[LSTM / GRU]
    RNN_F --> RNN3[Bi-LSTM]
    RNN_F --> RNN4[Seq2Seq + Attention]

    TRANS --> TRANS_E[Encoder-only]
    TRANS --> TRANS_D[Decoder-only]
    TRANS --> TRANS_ED[Encoder-Decoder]
    TRANS --> TRANS_V[Vision Transformers]

    TRANS_E --> T1[BERT]
    TRANS_E --> T2[RoBERTa / DistilBERT]
    TRANS_E --> T3[DeBERTa / ALBERT]

    TRANS_D --> T4[GPT-2 / GPT-3 / GPT-4]
    TRANS_D --> T5M[LLaMA 2 / 3]
    TRANS_D --> T6[Mistral / Mixtral]
    TRANS_D --> T7[Gemma / Falcon / Phi-3]

    TRANS_ED --> T8[T5 / FLAN-T5]
    TRANS_ED --> T9[BART / mBART]
    TRANS_ED --> T10[MarianMT / M2M-100]

    TRANS_V --> T11[ViT / DeiT]
    TRANS_V --> T12[Swin Transformer]
    TRANS_V --> T13[CLIP]

    GEN --> GEN_GAN[GAN Family]
    GEN --> GEN_DIFF[Diffusion Family]
    GEN --> GEN_VAE[VAE Family]

    GEN_GAN --> G1[DCGAN]
    GEN_GAN --> G2[StyleGAN 2 / 3]
    GEN_GAN --> G3[CycleGAN / Pix2Pix]

    GEN_DIFF --> G4[DDPM]
    GEN_DIFF --> G5[Stable Diffusion]
    GEN_DIFF --> G6[DALL-E 3 / Imagen]
    GEN_DIFF --> G7[ControlNet]

    GEN_VAE --> G8[VAE / β-VAE]
    GEN_VAE --> G9[VQ-VAE]

    GNN --> GNN1[GCN]
    GNN --> GNN2[GAT]
    GNN --> GNN3[GraphSAGE]
    GNN --> GNN4[GIN / MPNN]

    CNN_F --> CV_TASKS[CV Tasks]
    TRANS_V --> CV_TASKS

    CV_TASKS --> CV1[Image Classification - ResNet · EfficientNet · ViT]
    CV_TASKS --> CV2[Object Detection - YOLOv8 · Faster R-CNN · DETR]
    CV_TASKS --> CV3[Segmentation - U-Net · SAM · SegFormer]
    CV_TASKS --> CV4[Pose Estimation - OpenPose · MediaPipe · HRNet]
    CV_TASKS --> CV5[Tracking - DeepSORT · ByteTrack]
    CV_TASKS --> CV6[Depth Estimation - MiDaS · Depth Anything]

    TRANS_E --> NLP_TASKS[NLP Tasks]
    TRANS_D --> NLP_TASKS

    NLP_TASKS --> N1[Text Classification - BERT · RoBERTa · SetFit]
    NLP_TASKS --> N2[NER / IE - BERT-NER · spaCy · Flair]
    NLP_TASKS --> N3[Question Answering - BERT-SQuAD · RAG · DPR]
    NLP_TASKS --> N4[Summarization - BART · PEGASUS · T5]
    NLP_TASKS --> N5[Translation - MarianMT · NLLB-200]
    NLP_TASKS --> N6[Embeddings - Sentence-BERT · E5 · BGE]
    NLP_TASKS --> N7[Speech / ASR - Whisper · wav2vec · HuBERT]

    TRANS_D --> LLM[LLM + RAG Production Stack]
    LLM --> LLM1[Prompt Engineering]
    LLM --> LLM2[Fine-tuning / LoRA]
    LLM --> LLM3[RAG + Vector DB]
    LLM --> LLM4[Agents / Tool Use]

    CV_TASKS --> MULTI([🌍 Multimodal AI])
    NLP_TASKS --> MULTI
    LLM --> MULTI
    GEN --> MULTI

    MULTI --> M1[GPT-4o]
    MULTI --> M2[Gemini 1.5]
    MULTI --> M3[Claude 3.5]
    MULTI --> M4[LLaVA / BLIP-2]
    MULTI --> M5[Sora / Gen-3]

    classDef root fill:#1e293b,stroke:#64748b,color:#f1f5f9,font-weight:bold
    classDef trad fill:#1e3a5f,stroke:#3b82f6,color:#93c5fd
    classDef dl fill:#1e3b2f,stroke:#22c55e,color:#86efac
    classDef nlp fill:#2d1b4e,stroke:#a78bfa,color:#c4b5fd
    classDef cv fill:#3b1f0f,stroke:#f97316,color:#fed7aa
    classDef gen fill:#2d1b2e,stroke:#ec4899,color:#fbcfe8
    classDef multi fill:#2d2614,stroke:#fbbf24,color:#fde68a
    classDef rl fill:#2d1515,stroke:#f87171,color:#fecaca
    classDef leaf fill:#111827,stroke:#374151,color:#9ca3af

    class AI,ML root
    class TRAD,SUP,UNSUP,REG,CLF,ENS,CLU,DIM,ANO trad
    class REG1,REG2,REG3,REG4,CLF1,CLF2,CLF3,CLF4,CLF5 leaf
    class ENS1,ENS2,ENS3,ENS4,ENS5,ENS6 leaf
    class CLU1,CLU2,CLU3,CLU4,DIM1,DIM2,DIM3,DIM4 leaf
    class ANO1,ANO2,ANO3,ANO4 leaf
    class DL,ANN,CNN_F,RNN_F,TRANS,GEN,GNN dl
    class ANN1,ANN2,CNN1,CNN2,CNN3,CNN4 leaf
    class RNN1,RNN2,RNN3,RNN4 leaf
    class TRANS_E,TRANS_D,TRANS_ED,TRANS_V nlp
    class T1,T2,T3,T4,T5M,T6,T7,T8,T9,T10,T11,T12,T13 leaf
    class GEN_GAN,GEN_DIFF,GEN_VAE gen
    class G1,G2,G3,G4,G5,G6,G7,G8,G9 leaf
    class GNN1,GNN2,GNN3,GNN4 leaf
    class RL,RL_D,RL_C,RL_M rl
    class RL1,RL2,RL3,RL4,RL5,RL6,RL7,RL8,RL9 leaf
    class CV_TASKS,CV1,CV2,CV3,CV4,CV5,CV6 cv
    class NLP_TASKS,N1,N2,N3,N4,N5,N6,N7 nlp
    class LLM,LLM1,LLM2,LLM3,LLM4 nlp
    class MULTI,M1,M2,M3,M4,M5 multi
    ```
