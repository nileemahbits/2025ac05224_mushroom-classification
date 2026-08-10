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
- Naive Bayes (Multinomial)
- Random Forest (Ensemble)



# Comparison Table with the evaluation metrics calculated for all the models

| ML Model Name       | Accuracy | AUC     | Precision | Recall  | F1      | MCC     |
|---------------------|----------|---------|-----------|---------|---------|---------|
| Logistic Regression | 1.0000   | 1.0000  | 1.0000    | 1.0000  | 1.0000  | 1.0000  |
| kNN                 | 0.9988   | 1.0000  | 1.0000    | 0.9974  | 0.9987  | 0.9975  |
| Decision Tree       | 1.0000   | 1.0000  | 1.0000    | 1.0000  | 1.0000  | 1.0000  |
| Naive Bayes         | 0.9458   | 0.9972  | 0.9901    | 0.8966  | 0.9410  | 0.8949  |
| Random Forest       | 1.0000   | 1.0000  | 1.0000    | 1.0000  | 1.0000  | 1.0000  |



# Observations on model performance

| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | Achieved perfect classification metrics on the test set with a perfect AUC, confirming that the one-hot encoded and scaled features are highly separable for this dataset. Training is efficient (~0.21s). No overfitting observed (0.00 gap).|
| Decision Tree       | Also achieved perfect metrics, indicating the tree can fully partition the categorical feature space with no misclassifications on this test split. Extremely fast (~0.03s) and generates clear logical rules. No overfitting observed(0.00 gap).|
| kNN                 | Delivered near-perfect accuracy and perfect precision, with only a single false negative causing recall to drop slightly below 1.0. Fastest training time (~0.007s).Generalizes well with a negligible gap (0.0009). |
| Naive Bayes         | Showed the weakest overall performance, with lower recall and F1 than the tree-based models, reflecting its conditional independence assumptions on categorical features. Fast training (~0.018s) with a minor generalization gap (0.0086). |
| Random Forest (Ensemble)| Achieved perfect classification metrics and is the strongest candidate for deployment, offering the robustness of an ensemble while matching the decision tree’s performance. Slowest training (~0.31s) due to complexity. No overfitting observed(0.00 gap).|
| **Overall Winner for the dataset?** | Logistic Regression and Decision Trees models provide a 100% reliable classification for this dataset with zero evidence of overfitting.Overall winner will be , the **Decision Tree** because it is optimal as it matches perfect accuracy with high speed and human-readable logic.|


