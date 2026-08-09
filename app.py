import pickle

import streamlit as st
import pandas as pd
from io import StringIO

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score




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
   
    ascore=accuracy_score(y_test, y_predict)
    pscore=precision_score(y_test, y_predict)
    rscore=recall_score(y_test, y_predict)
    f1=f1_score(y_test, y_predict)

    st.write(f"Accuracy Score: {ascore}")
    st.write(f"Precision Score: {pscore}")
    st.write(f"Recall Score: {rscore}")
    st.write(f"F1 Score: {f1}")
else:
    st.write("Please upload a test data file and select a model to see evaluation metrics.")
