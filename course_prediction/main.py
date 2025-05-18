# import essential libraries
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import streamlit as st 

# Load and preprocess the dataset
df=pd.read_csv("student_data.csv",encoding="latin1")
df.columns=df.columns.str.strip().str.replace(" ","_").str.replace("-","_").str.replace(".","_")
df['College']=df['College'].fillna('Not provided')
df['College']=df['College'].str.strip().str.title()
df['Branch']=df['Branch'].str.strip().str.title()
df['Course']=df['Course'].str.strip().str.title()
df['Subject']=df['Subject'].str.strip().str.title()

#Encode features and train ML model for prediction
features=['Branch','College','Course','Year']
target='Subject'
df_ml=df.dropna(subset=features+[target])
encoders={}
for col in features+[target]:
    le=LabelEncoder()
    df_ml[col]=le.fit_transform(df_ml[col])
    encoders[col]=le
X=df_ml[features]
y=df_ml[target]
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X,y)


# Prediction function using ml model
def predict_subject_ml(branch,college,course,year,top_n=3):
    input_df=pd.DataFrame([[branch,college,course,year]],columns=features)
    for col in features:
        input_df[col]=encoders[col].transform(input_df[col])
         
    probs=model.predict_proba(input_df)[0]
    top_indices=np.argsort(probs)[::-1][:top_n]
    subject_names=encoders[target].inverse_transform(top_indices)
    return list(zip(subject_names,probs[top_indices]))



# Streamlit UI
st.title("Course Recommendation system")
st.markdown("Get top reommended course based on your branch,college,course and year")

#sidebar input
branches=sorted(df['Branch'].dropna().unique())
colleges=sorted(df['College'].dropna().unique())
courses=sorted(df['Course'].dropna().unique())
years=sorted(df['Year'].dropna().unique())

selected_branch=st.selectbox("select your branch",branches)
selected_college=st.selectbox("select your college",colleges)
selected_course=st.selectbox("select your course",courses)
selected_years=st.selectbox("select your year",years)
button=st.button("Recommended courses(ML-Based)")
if button:
    ml_recommendations=predict_subject_ml(selected_branch,selected_college,selected_course,selected_years)
    st.subheader("ML based recommended subjects")
    for i,(subject,score)in enumerate(ml_recommendations):
        st.markdown(f"{i}.**{subject}**- Confidence:{score:.2f}")