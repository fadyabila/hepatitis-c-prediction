import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Choose Page : ', ('EDA', 'Patients Diagnose'))

if navigation == 'EDA':
    eda.run()
else:
    prediction.run()