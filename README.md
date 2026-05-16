# Wearable Sensors and MET Features: Human Activity Recognition

Repository for the research paper: **Wearable Sensors and MET Features: The importance of Preprocessing and Understanding the Data**. This repository contains the raw/processed datasets, Python algorithms for the analysis, and the LaTeX source for the paper.

## Repository Structure

```
├── Figures/                 # EPS plots and figures referenced in the paper
├── src/                     # Python scripts for data preprocessing and ML classification
└── data/                    # Processed and organized sensor datasets (grouped by subjects F-01, M-01, etc.)
```

## Setup & Usage

### 1. Data Processing and Classification
The `src/` directory contains the Python scripts to load the data, apply highpass filtering to separate gravitational components, and train multiple Machine Learning models (Logistic Regression, LDA, KNN, DT-CART, Naive Bayes) including Deep Learning alternatives (TabNet).

- `MLAlgoritmos.py` / `MLAlgoritmosGroupByActivities.py`: Main execution loops for classical models.
- `ML_boxplots.py`: Generates the accuracy visualization plots.
- `har.py`: General helper functions for signal preprocessing.

## Main Conclusions

This study demonstrates that incorporating metabolic-equivalent (MET) features—such as the filtered acceleration magnitude ($SVMg_{fil}$), the Ratio of Unfiltered to Filtered acceleration (RUF), and the percentage of Heart Rate Reserve (%HRR)—significantly improves the accuracy of Human Activity Recognition (HAR) models. 

These features provide a more robust representation of movement and physiological data. As our results show, combining these metrics leads to accuracy improvements of up to 166.67%, allowing for better differentiation of challenging, high-intensity activities (like stair climbing and household chores) which are difficult to distinguish using only raw acceleration data.

By integrating heart rate data with acceleration patterns, this approach contributes to developing more reliable HAR systems with potential applications in healthcare, fitness tracking, and everyday activity monitoring.

![Model Accuracies using MET Features](Figures/all-smvRUFHRR.eps)
