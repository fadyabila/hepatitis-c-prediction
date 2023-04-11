import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Load All Files

with open('pipelines.pkl', 'rb') as file_1:
  pipelines = pickle.load(file_1)

with open('num_columns.txt', 'r') as file_2:
  num_columns = json.load(file_2)

with open('cat_columns.txt', 'r') as file_3:
  cat_columns = json.load(file_3)

with open('norm_columns.txt', 'r') as file_4:
  norm_columns = json.load(file_4)

with open('skew_columns.txt', 'r') as file_5:
  skew_columns = json.load(file_5)

with open('enc_columns.txt', 'r') as file_6:
  enc_columns = json.load(file_6)

def run():
    with st.form(key='Hepatitis_C_Prediction'):
        Category = st.selectbox('Category', ('0=Blood Donor', '1=suspect Blood Donor', 
                                             '2=Hepatitis', '3=Fibrosis', '4=Chirrosis'), index=1)
        Age = st.number_input('Age', min_value=23, max_value=80, value=23)
        Sex = st.selectbox('Sex', ('Male', 'Female'), index=1)
        st.markdown('---')

        ALB = st.number_input('ALB', min_value=0, max_value=85, value=0)
        ALP = st.number_input('ALP', min_value=0, max_value=420, value=0)
        ALT = st.number_input('ALT', min_value=0, max_value=325, value=0)
        AST = st.number_input('AST', min_value=0, max_value=325, value=0)
        BIL = st.number_input('BIL', min_value=0, max_value=210, value=0)

        st.markdown('---')

        CHE = st.number_input('CHE', min_value=0, max_value=20, value=0)
        CHOL = st.number_input('CHOL', min_value=0, max_value=10, value=0)
        CREA = st.number_input('CREA', min_value=0, max_value=1080, value=0)
        GGT = st.number_input('GGT', min_value=0, max_value=651, value=0)
        PROT = st.number_input('PROT', min_value=0, max_value=87, value=0)

        submitted = st.form_submit_button('Predict')

    data_inf = {
    'Category': Category,
    'Age': Age,
    'Sex': Sex,
    'ALB': ALB,
    'ALP': ALP,
    'ALT': ALT,
    'AST': AST,
    'BIL': BIL,
    'CHE': CHE,
    'CHOL': CHOL,
    'CREA': CREA,
    'GGT': GGT,
    'PROT': PROT
    }

    data_inf = pd.DataFrame([data_inf])    
    st.dataframe(data_inf)

    if submitted:

        # Predict using Linear Regression
        y_pred_inf = pipelines.predict(data_inf)
        st.write('# Diagnose : ', str(int(y_pred_inf)))

if __name__ == '__main__':
    run()