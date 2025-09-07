# Student Performance Prediction and Analytics Dashboard

# Required Libraries
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Title
st.title("🎓 Student Performance Prediction and Analytics Dashboard")

# Upload CSV File
uploaded_file = st.file_uploader("Upload Student Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Dataset")
    st.dataframe(df)

    # Handle missing values
    df = df.dropna()

    # Basic statistics
    st.subheader("📈 Dataset Summary")
    st.write(df.describe())

    # Correlation Heatmap
    st.subheader("📉 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 5))
    numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
    if not numeric_df.empty:
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for correlation heatmap.")

    # Feature Selection
    st.subheader("⚙️ Select Features for Prediction")
    all_columns = df.columns.tolist()
    target = st.selectbox("Select Target Column (e.g., 'Result')", all_columns)
    features = st.multiselect("Select Feature Columns", [col for col in all_columns if col != target])

    if target and features:
        # Prepare input and output
        X = df[features].copy()
        y = df[target].copy()

        # Encode non-numeric features
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col])

        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        else:
            le = None

        # Model Training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

        # Confusion Matrix
        st.subheader("📌 Confusion Matrix")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Real-time Prediction
        st.subheader("🔮 Predict a New Student Result")
        input_data = []
        
        
for col in features:
    if df[col].dtype == 'object':
        options = df[col].unique().tolist()
        value = st.selectbox(f"Select {col}", options)
        # Encode string input to numeric
        le_feature = LabelEncoder()
        le_feature.fit(df[col])
        value = le_feature.transform([value])[0]
    else:
        value = st.number_input(f"Enter {col}", value=0.0)
    input_data.append(value)


    if st.button("Predict"):
            prediction = model.predict([input_data])[0]
            if le:
                prediction = le.inverse_transform([prediction])[0]
            st.success(f"Predicted Result: {prediction}")

else:
    st.info("👆 Please upload a CSV file to begin.")

# Example CSV structure (optional display or download)
with st.expander("📄 Example CSV Format"):
    st.code("""StudentID,Attendance,AssignmentScore,InternalMarks,Result
101,85,80,75,Pass
102,60,50,45,Fail
103,90,85,88,Pass
""", language='csv')
