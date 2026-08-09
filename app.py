import pickle

import streamlit as st
import pandas as pd
from io import StringIO

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def model_performance_analysis(score_matrix,cm):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")

        # Define custom labels
        class_labels = ["Edible", "Poisonous"]

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,  
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
            ax=ax,
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

     
        st.pyplot(fig)

    # --- COLUMN 2: METRICS TABLE ---
    with col2:
        st.subheader("Performance Scores")
        labels = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "AUC Score",
            "Matthews Corr (MCC)",
        ]

        # Convert to a DataFrame and format to 4 decimal places
        df_scores = pd.DataFrame({"Score": score_matrix}, index=labels)
        df_scores["Score"] = df_scores["Score"].map("{:.4f}".format)

        # Render a clean, static HTML table in Streamlit
        st.table(df_scores)

model_map = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "kNN": "knn_model.pkl",
    "Naive Bayes": "naive_bayes_gaussian_model.pkl",
    "Random Forest": "random_forest_model.pkl"
}
 
st.write("""
# Mushroom Classification app
""")


X_test = None
y_test = None
y_predict = None
loaded_model = None


uploaded_file = st.file_uploader("Choose a test data file")
if uploaded_file is not None:
    # To read file as bytes:
    # bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)
    string_data = stringio.read()

    df = pd.read_csv(StringIO(string_data))

    X = df.drop('class', axis=1)
    y_test = df['class']
    X = pd.get_dummies(X, drop_first=True)

    X_test = X

    le = LabelEncoder()
    y_test = le.fit_transform(y_test)



if X_test is not None:

    option = st.selectbox(
        "Select Model",
        ("Logistic Regression", "Decision Tree", "kNN", "Naive Bayes","Random Forest")
    )

    # Display the selected option
    st.write("You selected:", option)

    if option == "Logistic Regression" or option == "kNN":
        with open('./model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            X_test = scaler.transform(X_test)
     
    filename = "./model/" + model_map.get(option)

    with open(filename, "rb") as file:
        loaded_model = pickle.load(file)
        y_predict = loaded_model.predict(X_test)        
        st.write(f"Predictions: {y_predict}")


st.write("## Model Evaluation Metrics")

if y_predict is not None:

    accuracy=accuracy_score(y_test, y_predict)
    precision=precision_score(y_test, y_predict)
    recall=recall_score(y_test, y_predict)
    f1=f1_score(y_test, y_predict)     
    y_probs = loaded_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_probs)    
    mcc=matthews_corrcoef(y_test, y_predict)
    
    score_matrix = [accuracy, precision, recall, f1, auc, mcc]
    cm = confusion_matrix(y_test, y_predict)

    model_performance_analysis(score_matrix, cm)
   
else:
    st.write("Please upload a test data file and select a model to see evaluation metrics.")
