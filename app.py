import pickle

import streamlit as st
import pandas as pd
from io import StringIO

from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def model_performance_analysis(score_matrix,cm):

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

    st.subheader("Confusion Matrix")
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


X_raw = None
y_test = None
y_predict = None
loaded_model = None


uploaded_file = st.file_uploader("Choose a test data file")
if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    string_data = stringio.read()

    df = pd.read_csv(StringIO(string_data))

    X_raw = df.drop('class', axis=1)
    y_test = df['class']

    with open('./model/trained_models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
        y_test = le.transform(y_test)
    

if X_raw is not None:
    option = st.selectbox(
        "Select ML classification Model",
        ("Decision Tree","Logistic Regression","Random Forest", "kNN", "Naive Bayes",)
    )

    # Display the selected option
    st.write("You selected:", option)

    if option in ("Logistic Regression", "kNN", "Naive Bayes"):
        with open('./model/trained_models/encoder.pkl', 'rb') as f:
            encoder = pickle.load(f)
            X_test = encoder.transform(X_raw)

        if option in ("Logistic Regression", "kNN"):
            with open('./model/trained_models/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
                X_test = scaler.transform(X_test)
    else:
        X_test = X_raw.apply(lambda x: x.astype('category').cat.codes)
     
    filename = "./model/trained_models/" + model_map.get(option)

    with open(filename, "rb") as file:
        loaded_model = pickle.load(file)
        y_predict = loaded_model.predict(X_test)



if y_predict is not None:
    st.write(f"## {option} Evaluation Metrics")

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

    ##########################################################################
    # Add predictions back to the original dataframe and save for download
    df["y_predict"] = y_predict
    try:
        df["predicted_class"] = le.inverse_transform(y_predict)
    except Exception:
        pass

    output_path = "./predictions.csv"
    df.to_csv(output_path, index=False)

    csv = df.to_csv(index=False)
    st.download_button(
        label="Download predictions CSV",
        data=csv,
        file_name="mushroom_predictions.csv",
        mime="text/csv",
    )
    st.write(f"Saved predictions file to `{output_path}`")
    ##########################################################################
   
else:
    st.write("Please upload a test data file and select a model to see evaluation metrics.")
