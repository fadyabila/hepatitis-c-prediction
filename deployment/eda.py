import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

# Melebarkan visualisasi untuk memaksimalkan browser
st.set_page_config(
    page_title='Hepatitis C',
    layout='wide',
    initial_sidebar_state='expanded'
)

def run():
    # Membuat title
    st.title('Patients with Hepatitis C Diagnose')
    st.write('### by Fadya Ulya Salsabila')

    # Menambahkan Gambar
    image = Image.open('hepatitis.jpg')
    st.image(image, caption='Stages of Hepatitis C Disease')

    # Menambahkan Deskripsi
    st.write('## Background')
    st.write(""" Hepatitis C is a very dangerous and contagious disease, which is caused by the Hepatitis C virus (HCV). 
    Transmission of Hepatitis C can be through body fluids, blood, or when having sex with sufferers. 
    Usually, the symptoms of Hepatitis C are not visible, so suddenly the sufferer or patient has experienced the chronic stage of hepatitis. 
    Hepatitis C can also trigger the onset of fibrosis which causes cirrhosis and liver cancer.

    Therefore, the Liver Hospital must have adequate facilities and treatment to treat patients with indications and symptoms of Hepatitis C. 
    The purpose of this data classification analysis and modeling is to find out what the diagnoses are in patients so that they receive further treatment, can be cured, and minimize death. 
    This modeling uses 4 Machine Learning (Supervised) algorithms, namely Logistic Regression, Support Vector Machine, Random Forest, and Gradient Boosting to get the best predictions with hyperparameter tuning.""")

    st.write('## Dataset')
    st.write("""
    The dataset is from Kaggle, that have 615 rows and 14 columns.
    This column contains information that have a correlation with Hepatitis C disease, such as:
    1. Bilirubin
    2. Protein and Albumin
    3. Creatinine. Elevation in serum creatinine is a common laboratory finding for patients with cirrhosis and can indicate the presence of either an acute kidney injury (AKI) or chronic kidney disease (CKD).
    4. Alanine transaminase (ALT): ALT is an enzyme found in the liver that helps convert proteins into energy for the liver cells. When the liver is damaged, ALT is released into the bloodstream and levels increase.
    5. Aspartate transaminase(AST): AST is an enzyme that helps metabolize amino acids. Like ALT, AST is normally present in blood at low levels. An increase in AST levels may indicate liver damage, disease or muscle damage.
    6. Alkaline phosphatase (ALP): ALP is an enzyme found in the liver and bone and is important for breaking down proteins. Higher-than-normal levels of ALP may indicate liver damage or disease, such as a blocked bile duct, or certain bone diseases.
    7. Albumin and total protein: Albumin is one of several proteins made in the liver. Your body needs these proteins to fight infections and to perform other functions. Lower-than-normal levels of albumin and total protein may indicate liver damage or disease.
    8. Gamma-glutamyltransferase (GGT): GGT is an enzyme in the blood. Higher-than-normal levels may indicate liver or bile duct damage.""")

    # Membuat Garis Lurus
    st.markdown('---')

    # Membuat Sub Headrer
    st.subheader('EDA for Analyze Patients')

    # Magic Syntax
    st.write(
    'On this page, the author will do a simple exploration.'
    ' The dataset used is the Hepatitis C dataset.'
    ' This dataset comes from Kaggle.')

    # Show DataFrame
    df1 = pd.read_csv('HepatitisCdata.csv')
    st.dataframe(df1)

    # Membuat Barplot
    st.write('#### Diagnose Plot')
    fig = plt.figure(figsize=(10,7))
    sns.countplot(x='Category', data=df1, palette="PuRd")
    st.pyplot(fig)

    st.write('#### Sex Based on Diagnose')
    fig1, ax1 = plt.subplots(figsize=(15, 8))
    sns.countplot(x='Sex', hue='Category', data=df1, ax=ax1)
    st.pyplot(fig1)

    # Mengelompokkan Usia
    bins = [18, 30, 40, 50, 60, 70, 120]
    labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-80']
    df1['agerange'] = pd.cut(df1.Age, bins, labels = labels,include_lowest = True)

    # Menampilkan visualisasi usia berdasarkan diagnosis
    st.write('#### Age Based on Diagnose')
    fig2, ax2 = plt.subplots(figsize=(10,7))
    sns.countplot(x='agerange', data=df1, hue="Category", ax=ax2)
    st.pyplot(fig2)

    # Rata-rata AST berdasarkan diagnosis
    AST = df1.groupby('Category').agg({'AST':'mean'}).reset_index()
    AST = AST.sort_values(by='AST')
    # Menampilkan visualisasi bar chart
    st.write('#### AST Mean Based on Diagnose')
    fig3, ax3 = plt.subplots(figsize=(10,7))
    sns.barplot(data=AST, x=AST.Category, y=AST.AST, ax=ax3)
    st.pyplot(fig3)

    # Rata-rata CHE berdasarkan diagnosis
    CHE = df1.groupby('Category').agg({'CHE':'mean'}).reset_index()
    CHE = CHE.sort_values(by='CHE')
    # Menampilkan visualisasi bar chart
    st.write('#### CHE Mean Based on Diagnose')
    fig4, ax4 = plt.subplots(figsize=(10,7))
    sns.barplot(data=CHE, x=CHE.Category, y=CHE.CHE, ax=ax4)
    st.pyplot(fig4)

    # Membuat heatmap correlation
    st.write('#### Heatmap Correlation')
    fig = plt.figure(figsize = (15,8))
    sns.heatmap(df1.corr(), annot = True, square = True)
    st.pyplot(fig)

    # Membuat Histogram Berdasarkan Input User
    st.write('#### Histogram Based On User Input')
    pilihan = st.selectbox('Choose Column : ', ('Age', 'Sex', 'AST', 
                                                'CHE', 'BIL', 'GGT'))
    fig = plt.figure(figsize=(15,5))
    sns.histplot(df1[pilihan], bins=30, kde=True)
    st.pyplot(fig)

if __name__ == '__main__':
    run()