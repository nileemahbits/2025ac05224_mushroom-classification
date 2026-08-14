# Mushroom Classification

# a. Problem statement

The task is to build a machine learning classifier that predicts whether a mushroom is edible or poisonous using the Mushroom Classification dataset. The goal is to compare several supervised learning algorithms and select the best-performing model for this binary classification problem.
It also requires to build an interactive Streamlit web application to demonstrate the trained models.


# b. Dataset description

The dataset used in this project is the Mushroom Classification dataset from Kaggle. The dataset contains characteristics of different mushrooms, with the primary objective of classifying mushrooms according to their edibility. The target contains two classes,Edible and Poisonous, hence, this is a binary class
- Dataset Source: Kaggle — Mushroom Classification Dataset
- Dataset link : https://www.kaggle.com/datasets/uciml/mushroom-classification
- Dataset Characteristics: The dataset contains categorical features describing different physical characteristics of mushrooms. 
    - Examples of such characteristics include: Cap shape, Cap surface, Cap color, Bruises, Odor, Gill characteristics, Stalk characteristics, Habitat, Population and Other mushroom physical attributes
- Feature type: Primarily categorical
- Classification type: Binary Classification
- Target classes:
  - Edible
  - Poisonous
- Number of instances: 8,124
- Number of features: 22




# c. Github Repository Link

https://github.com/nileemahbits/2025ac05224_mushroom-classification.git

# d. Models used

The following models were trained and evaluated on this dataset:
- Logistic Regression
- k-Nearest Neighbors (kNN)
- Decision Tree
- Naive Bayes (Multinomial)
- Random Forest (Ensemble)

The models were evaluated using:
- Accuracy
- AUC
- Precision
- Recall
- F1-Score
- Matthews Correlation Coefficient (MCC)

Since the dataset primarily contains categorical features, CategoricalNB
would generally be the most suitable choice. However, as it was not available
among the assignment's model options, the choice was between GaussianNB
and MultinomialNB. GaussianNB is better suited for continuous numerical
features, while MultinomialNB works well with discrete, non-negative features
such as counts or frequencies. Therefore, the categorical features were one-hot
encoded into binary, non-negative features, making MultinomialNB the more
appropriate choice among the available options.

The models were evaluated using: Accuracy , AUC ,Precision , Recall , F1-
Score , Matthews Correlation Coefficient (MCC). Both training and testing
performance were considered to identify generalization and possible overfitting.

## Comparison Table with the evaluation metrics calculated for all the models
The following table summarizes the test-set performance, which is the most important measure for comparing how well the models generalize to unseen data

| ML Model Name       | Accuracy | AUC     | Precision | Recall  | F1      | MCC     |
|---------------------|----------|---------|-----------|---------|---------|---------|
| Logistic Regression | 1.0000   | 1.0000  | 1.0000    | 1.0000  | 1.0000  | 1.0000  |
| kNN                 | 0.9988   | 1.0000  | 1.0000    | 0.9974  | 0.9987  | 0.9975  |
| Decision Tree       | 1.0000   | 1.0000  | 1.0000    | 1.0000  | 1.0000  | 1.0000  |
| Naive Bayes (Multinomial)         | 0.9458   | 0.9972  | 0.9901    | 0.8966  | 0.9410  | 0.8949  |
| Random Forest (Ensemble)     | 1.0000   | 1.0000  | 1.0000    | 1.0000  | 1.0000  | 1.0000  |

**Train-test Gaps**: The very small train-test gaps(as seen in the notebook file) indicate that the models generally maintain their performance on unseen test data. Naive Bayes has the largest gap, although the gap of approximately 0.86 percentage points is still relatively small.

**Training Time Observation**: kNN required the least measured training time, followed by Naive Bayes and Decision Tree. Random Forest required more training time because it builds an ensemble of multiple decision trees. Logistic Regression had the highest measured training time among the five models in this particular experiment.


## Observations on model performance

| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | Logistic Regression performed exceptionally well. All six metrics were exactly 1.00 on both training and test data. The absence of a train-test gap suggests that the model generalized perfectly to the provided test set.|
| Decision Tree       | Decision Tree achieved the same perfect results as Logistic Regression. Its ability to learn nonlinear decision boundaries and categorical feature relationships makes it particularly suitable for this dataset. The model also had a very low measured training time.|
| kNN                 | KNN performed almost perfectly, with a test accuracy of 99.877%. Its Precision remained perfect, while its Recall was slightly below 1.00. The very small train-test gap demonstrates that KNN generalizes extremely well for the selected dataset split.|
| Naive Bayes(Multinomial)         | Naive Bayes showed good but noticeably lower performance compared with the other models. Its test accuracy was 94.585%. Despite this, its AUC of 99.718% demonstrates excellent ranking/discrimination capability. The relatively lower Recall contributed to its lower F1 and MCC scores.|
| Random Forest (Ensemble)| Random Forest achieved perfect scores across every evaluated metric. As an ensemble method, it combines predictions from multiple decision trees and is generally capable of modeling complex relationships in tabular datasets. In this experiment, it performed perfectly on both the training and test sets.|


## Overall Winner for the dataset?

 **Decision Tree** 
 - The three top-performing models — Logistic Regression, Decision Tree, and Random Forest — achieved identical perfect results on the test set, with 1.00 Accuracy, AUC, Precision, Recall, F1-Score, and MCC.
- To select a single overall winner, training time is considered as a tie-breaking factor. Logistic Regression(0.204874 sec) and Random Forest(0.097327 sec) required substantially more training time than Decision Tree(0.004762 sec) in the measured experiment.
- The Decision Tree achieved the same perfect predictive performance as Logistic Regression and Random Forest while requiring only 0.004762 seconds of training time.
- Therefore, when considering both predictive performance and training efficiency, Decision Tree is selected as the practical overall winner for this particular experiment. 
- It provides perfect classification performance on the test set with significantly lower training time, making it the most efficient choice among the top-performing models.
