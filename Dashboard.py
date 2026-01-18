import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import pickle


st.set_page_config(page_title="Student Score Predictor",
                   layout='wide',
                   initial_sidebar_state='expanded')         

st.markdown("""
<style>

/* ====== Main App Background Soft Gray ====== */
.stApp {
    background-color: #e0e0e0; /* Soft gray */
    color: #1f2933; /* Dark text */
    font-family: 'Segoe UI', 'Roboto', sans-serif;
}

/* ====== Sidebar ====== */
section[data-testid="stSidebar"] {
    background-color: #f5f5f5; /* Light gray sidebar */
    border-right: 1px solid #d1d5db;
}

/* ====== Header Titles ====== */
h1, h2, h3 {
    color: #1f2933;
    font-weight: 600;
}

/* ====== Metric Cards ====== */
div[data-testid="metric-container"] {
    background-color: #ffffff; /* White card for contrast */
    border-radius: 12px;
    padding: 14px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    color: #1f2933;
}

/* ====== Buttons ====== */
.stButton>button {
    background-color: #6b7280;  /* Soft Gray-Blue */
    color: white;
    border-radius: 8px;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #4b5563;
}

/* ====== Inputs & Select Boxes ====== */
div[data-baseweb="select"],
input, textarea {
    background-color: #f0f0f0 !important;
    color: #1f2933 !important;
    border-radius: 6px !important;
    border: 1px solid #d1d5db !important;
}

/* ====== Plotly Background ====== */
.plotly-graph-div {
    background-color: #e0e0e0 !important;
    color: #1f2933 !important;
}

</style>
""", unsafe_allow_html=True)

# ====== Header Bar Soft Gray ======
st.markdown("""
<div style="
    width: 100%;
    background-color: #f5f5f5;
    padding: 12px 20px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    margin-bottom: 20px;
">
    <img src='https://upload.wikimedia.org/wikipedia/commons/a/ab/Logo_TV_2015.png' 
         style='height:36px; margin-right:15px; border-radius:5px;'/>
    <h1 style='color:#1f2933; font-size:22px; margin:0;'>Student Exam Score Predictor</h1>
</div>
""", unsafe_allow_html=True)





with open("model.pickle", "rb") as f:
    data = pickle.load(f)

model = data["model"]
encoders  = data["label_encoders"]
ohe_encoders = data["onehot_encoders"]
feature_cols = data["feature_cols"]



df = pd.read_csv('train.csv')

# Sidebar

st.sidebar.header('📊 Student Exam Score Prediction')
st.sidebar.image('images.jpg')
st.sidebar.write('This Dashboard is using Dataset From Kaggle for educational purpose')

st.sidebar.write(" ")
st.sidebar.write("Filter your data :")


cat_filter = st.sidebar.selectbox('Categorical Filter',
                                  ['gender','course','internet_access','sleep_quality','study_method','facility_rating','exam_difficulty'])

st.sidebar.write(" ")
st.sidebar.markdown('made by Eng.[Ahmed Taha](https://www.linkedin.com/in/ahmed-taha-2b3a38241/)')







# Body
a1,a2,a3,a4 = st.columns(4)
a1.metric('Max.Exam Score',df['exam_score'].max())
a2.metric('Min.Exam Score',df['exam_score'].min())
a3.metric('Max.Study Hours',df['study_hours'].max())
a4.metric('Min.Study Hours',df['study_hours'].min())



st.subheader('Distribution of Exam Scores')

plt.figure(figsize=(12,6))
sns.histplot(
    data=df,
    x='exam_score',
    hue=cat_filter,
    bins=30,
)

st.pyplot(plt)

c1, c2 = st.columns((5,5))

with c1:
    st.subheader(f'Count of {cat_filter}') 

    fig = px.histogram(
        df,
        x=cat_filter,
        color=cat_filter,
        text_auto=True,              
        title=f'Count Plot of {cat_filter}'  
    )

    fig.update_layout(
        xaxis_title=cat_filter,
        yaxis_title='Count',
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


with c2:
    st.subheader(f'Distribution of {cat_filter}')
    fig = px.pie(
        df,
        names=cat_filter,
        title=f'{cat_filter} Distribution'
    )
    st.plotly_chart(fig, use_container_width=True)

st.write(" ")
st.title("📊 Student Exam Score Prediction")

# inputs
gender = st.selectbox('Gender',["male", "female", "other"])
course = st.selectbox('Course',["b.sc","bca","b.com","ba","diploma"])
study_hours = st.slider('Study Hours',0.0,12.0,0.5)
attendance = st.slider('Class Attendance %',0.0,100.0,75.0)
internet = st.selectbox("Internet Access", ["yes","no"])
sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
sleep_quality = st.selectbox("Sleep Quality", ["poor","average","good"])
study_method = st.selectbox("Study Method", ["self-study","online videos","group study","coaching","mixed"])
facility = st.selectbox("Facility Rating", ["low","medium","high"])
difficulty = st.selectbox("Exam Difficulty", ["easy","moderate","hard"])


if st.button('Predict Score'):

    input_df = pd.DataFrame([{
        "gender": gender,
        "course": course,
        "study_hours": study_hours,
        "class_attendance": attendance,
        "internet_access": internet,
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "study_method": study_method,
        "facility_rating": facility,
        "exam_difficulty": difficulty
    }])

    # -------- Label Encoding --------
    for col, le in encoders.items():
        input_df[col] = le.transform(input_df[col])

    # -------- One-Hot Encoding --------
    for col, ohe in ohe_encoders.items():
        encoded = ohe.transform(input_df[[col]])
        encoded_df = pd.DataFrame(
            encoded,
            columns=ohe.get_feature_names_out([col])
        )
        input_df = pd.concat([input_df.drop(col, axis=1), encoded_df], axis=1)

    # -------- Column Alignment --------
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)

    # -------- Prediction --------
    pred = model.predict(input_df)[0]
    st.success(f"🎯 Expected Exam Score: {pred:.2f}")
