# Master_Thesis_Subtyping_HF

This Master Thesis deals with the subtyping of Heart Failure with Deep Learning and Cluster Methods

# Phenotyping 
As a Phenotyping Algorithm the following was used : https://www.phekb.org/phenotype/heart-failure-hf-differentiation-between-preserved-and-reduced-ejection-fraction

- **Create ICD and EF Cohort:** This Notebook is extracting the MRNs of patients which have an ICD Code and Ejection Fraction measures. 
- **Load Vocab of Clinical Notes:** This Notebook is loading and saving the dictionary of the MSDW Clinical Notes
- **Get All Clinical Notes:** This Notebook is extracting the Clinical Notes for the specified Cohort and is identifying the Notes with an HF Diagnosis.
- **Get Baseline Characteristics:** This Notebook is extracting the Baseline Characteristics for a specific Cohort. 
- **Get HF Onset:** This Notebook is identifying the Onset of HF for a patient. (First Diagnosis)


# Feature Extraction 

## Static Features

### Supervised Features

-  **Create Supervised Features:** This notebook can be used to convert Excel Sheets, which contain Diagnosis or medication Features, to FIBER Conditions.
- **Supervised Feature extraction Pipeline:** This Notebook is extracting predefined features for a cohort
-  **Merge and Data Preperation:** This Notebook is Merging  different Dataframes containing features of patients together.
- **Hospitalization and Ejection Fraction:** This Notebook is extracting Outcomes specific for HF Patients. 

### Unsupervised Features
- **Unsupervised Features extraction Pipeline:** This Notebook is extraction features for a specific cohort for different dimensions like Diagnosis or Medications.
- **Merge and Data Preperation:** This Notebook is Merging the different Dimensions and applies data preparation Methods. The User can choose between replacing all NaN Values with Zeros or doing simple Imputation
- **Clean unsupervised features:** This Notebook can be used to clean the unsupervised extracted features.For numerical features(Lab and Vital Sign) It is dropping features were a certain threshold of patients do not have this features and is replacing outlier values with NaN so that they can be later imputed Furthermore are measures for the same type of test out of different Systems like IBEX or EPIC are dropped.For Categorical Features(Medication Diagnosis and procedures) the user can choose which further features should be dropped, by entering the feature name.

## Timeseries
- **Extract Time Series Data:** With this notebook the TimeSeries for predefined FIBER Conditions can be extracted.
- **Time Series Convert Numeric and combine all:** This Notebook is dividing the numerical values of one feature in high and low values. At the end all dimension (Vitalsigns Diagnosis etc.) are combined.
- **Time Series Dictionary Sequence:** This Notebook is creating a Dictionary for the different concepts and is adding the new "terms" to the data frame. Additionaly are the sequences per patient and per patient per day created.
- **Learn Embeddings:** This Notebook is training word2Vec embeddings (cbow and Skipgram) and evaluates them.
- ** TimeSeries preparation LSTM: ** This Notebook is converting the timeseries data to the format which the LSTM autoencoder needs. This is done in the format per patient and per patient per day. 

# Aggregated Pipieline Configutations:

## Dataframe
- df_path: Path to dataframe (String)
- df_name: Name of dataframe (String)
- age_filter: Age over 90 is fixed to 90 (Boolean)
- drop_age: age will be not considered in the pipeline (Boolean)
- drop_gender: gender will be not considered in the pipeline (Boolean)
## Preprocessing
- scaler: Encoder for Categorical Columns:
    - num_scaler_name: 
        - StandardScaler
        - MinMaxScaler
    - cat_scaler_name:
        - BinaryEncoder 
        - OneHotEncoder
## Dimension Reduction Methods
- dim_red_method:
    - PPCA
    - ICA
    - PCA
        - check_pca: Calculating the Variance represented by the diffreent numbers of dimensions(Boolean)
    - KPCA
    - TSNE
    - SVD
    - LDA
    - PCA_TSNE
    - ICA_TSNE
    - AE
         - a_f_decoder: Activation Function of the decoder 
         - a_f_encoder: Activation Function of the encoder 
         - batchsize
         - epochs
             -optimizer
          - loss_function
    - AE_TSNE
    - UMAP
        - tune_umap: different configurations are tried out (Boolean)
        - umap_distace: Minimum Distance between the data points (Float)
        - umap_neighbours: Number of Neighbours (Float)
    
- dimension: number of dimensions the dataset should be reduced to 
## Clustering
- cluster_method: 
    - kmenas
    - hierarchical (AgglomerativeClustering)
- ellbow-method: True or false
- n_cluster: number of cluster that should be applied to the dataset
## Feature Evaluation
- anova: apply anova test on numerical features
- chi: apply chi test on categorical features
- top_features: Number of top features that should be printed out 
 
## General
- plotting: Plotting of Scatter plots (Boolean)

# LSTM autoencoder
- **Training of LSTM Notebook:** Notebook Version for Training the LSTM Autoencoder 
- ** Training of LSTM python Code:** Plain Code wich can be run in the background
- ** Evaluation of LSTMs:** Notebook, which is evaluating all trained LSTMs
