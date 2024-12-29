# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:06:55 2023

@author: Starchild
"""

import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np



# Title
st.header("RandomForest IESS prediction model")

#input
ast=st.sidebar.number_input("Fp2 kurt")
bilirubin=st.sidebar.number_input("F4 kurt")
INR=st.sidebar.number_input("T3 kurt")
WBC=st.sidebar.number_input("C4 kurt")
platelet_count=st.sidebar.number_input("O1 kurt")
creatinine=st.sidebar.number_input("F3 skew")
sodium=st.sidebar.number_input("C3 skew")
heart_rate=st.sidebar.number_input("O1 skew")
dbp=st.sidebar.number_input("Fp1 δ psd")
temperature=st.sidebar.number_input("T5 δ psd")
spo2=st.sidebar.number_input("T6 θ psd")
age=st.sidebar.number_input("Fp1 δ de")



with open('randomforest.pkl', 'rb') as f:
    clf = joblib.load(f)
    f.close()
# with open('data_max_12.pkl', 'rb') as f:
#     data_max = pickle.load(f)
# with open('data_min_12.pkl', 'rb') as f:
#     data_min = pickle.load(f)
with open('randomforest_explainer.pkl', 'rb') as f:
    explainer = joblib.load(f)
    f.close()


# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    # Store inputs into dataframe
    columns = ['Fp2 kurt', 'F4 kurt', 'T3 kurt', 'C4 kurt', 'O1 kurt', 'F3 skew', 'C3 skew', 'O1 skew',  'Fp1 δ psd', 'T5 δ psd', 'T6 θ psd', 'Fp1 δ de']
    X = pd.DataFrame([[INR,creatinine,bilirubin,WBC,sodium,platelet_count,temperature,dbp,ast,spo2,heart_rate,age]], 
                     columns =columns )
    st.write('data:')
    st.dataframe(X)
    # X = (X-data_min)/(data_max-data_min)
    # st.write('Normalized data:')
    # st.dataframe(X)
    # Get prediction
    prediction = clf.predict(X.values)
    pred=clf.predict_proba(X.values)[0][1]
    shap_values2 = explainer(X)
    # Output prediction
    
    st.text(f"The probability that this patient will respond to treatment is {pred}.")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = shap.plots.bar(shap_values2[0], max_display=12)
    st.pyplot(fig)

    fig = shap.force_plot(explainer.expected_value,
                shap_values2[0].values,
                X.iloc[0,:], matplotlib=True)
    st.pyplot(fig)
    
    
    
    
    
    
    
