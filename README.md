# a. Problem statement

The task is to build a machine learning classifier that predicts whether a mushroom is edible or poisonous using the Mushroom Classification dataset. The goal is to compare several supervised learning algorithms and select the best-performing model for this binary classification problem.

# b. Dataset description

The dataset is the Mushroom Classification dataset from Kaggle / UCI Machine Learning Repository. It contains 8,124 mushroom samples with 22 categorical features plus the target label. Each instance is labeled as either edible (e) or poisonous (p). The dataset is nearly balanced with 4,208 edible and 3,916 poisonous examples. Features describe physical mushroom characteristics such as cap shape, cap color, bruises, odor, gill size, stalk shape, veil color, ring type, spore print color, population, and habitat.

# c. Github Repository Link

https://github.com/nileemahbits/2025ac05224_mushroom-classification.git

# d. Models used

The following models were trained and evaluated on this dataset:
- Logistic Regression
- k-Nearest Neighbors (kNN)
- Decision Tree
- Naive Bayes (Gaussian)
- Random Forest (Ensemble)



# Comparison Table with the evaluation metrics calculated for all the models

| ML Model Name       | Accuracy | AUC     | Precision | Recall  | F1      | MCC     |
|---------------------|----------|---------|-----------|---------|---------|---------|
| Logistic Regression | 1.0000   | 0.9994  | 1.0000    | 1.0000  | 1.0000  | 1.0000  |
| kNN                 | 0.9988   | 0.9289  | 1.0000    | 0.9975  | 0.9987  | 0.9975  |
| Decision Tree       | 1.0000   | 1.0000  | 1.0000    | 1.0000  | 1.0000  | 1.0000  |
| Naive Bayes         | 0.9619   | 0.9951  | 0.9287    | 0.9975  | 0.9618  | 0.9262  |
| Random Forest       | 1.0000   | 1.0000  | 1.0000    | 1.0000  | 1.0000  | 1.0000  |



# Observations on model performance

| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | Achieved perfect classification metrics on the test set with almost optimal AUC, indicating the encoded features are linearly separable after preprocessing. |
| Decision Tree       | Also achieved perfect metrics, showing that the tree can perfectly split the categorical feature space for this dataset. |
| kNN                 | Near-perfect performance with a slightly lower AUC, suggesting it is strong but less calibrated on probability estimates than the tree-based models. |
| Naive Bayes         | Performed worst among the evaluated models due to the strong independence assumption, which is not fully satisfied by the dataset's categorical features. |
| Random Forest (Ensemble)| Achieved perfect metrics and is the preferred overall winner because the ensemble approach is generally more robust and reliable than a single decision tree. |
| Overall Winner for the dataset?   | Random Forest is the overall winner for this dataset because it combines many trees to reduce the risk of overfitting and provides stable, perfect performance on the evaluated test set.|


