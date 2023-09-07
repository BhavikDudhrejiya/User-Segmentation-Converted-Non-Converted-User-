import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
from yellowbrick.cluster import KElbowVisualizer

@st.cache_data()
def load_data():
    df = pd.read_csv('user_data.csv')
    df['Is_conversions'] = [1 if i!=0 else 0 for i in df['conversions']]
    df['engagement_time_seconds'] = df['engagement_time_seconds'].fillna(df['engagement_time_seconds'].mean())
    return df

@st.cache_data()
def top_n():
    df = pd.read_excel('./therapyworks_data/Top_N.xlsx')
    return df

def split_conversion_nonconversion(data):
    converted_data = data[data['Is_conversions']!=0]
    nonconverted_data = data[data['Is_conversions']==0]
    return converted_data, nonconverted_data

col = [
       'sessions', 
       'engaged_sessions', 
       'engagement_rate',
       'engagement_time_seconds', 
       'bounces', 
       'bounce_rate',
       'event_count_per_session', 
       'number_of_sessions_per_user', 
       'event_count',
       'event_count_per_user', 
       'engaged_sessions_per_user', 
       'page_views',
       'unique_pageviews', 
       'pages_per_session', 
       'pages_per_user', 
       'entrances'
       ]

def kmean_clustering(nonconverted_data):
    distortions = []
    K = range(1,15)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(nonconverted_data)
        distortions.append(kmeanModel.inertia_)
    df = pd.DataFrame()
    df['k'] = K
    df['WSS'] = distortions
    df['distance_from_mean'] = abs(df['WSS']-df['WSS'].mean())
    optimal_k = df['k'][df['distance_from_mean']==df['distance_from_mean'].min()].values[0]
    c1, c2 = st.columns(2)
    fig = px.line(df, x='k', y='WSS', markers=True)
    fig.add_vline(x=optimal_k, line_width=3, line_dash="dash", line_color="green")
    return df, optimal_k, c1.plotly_chart(fig)

def elbow_method(data):
    km = KMeans(random_state=42)
    visualizer = KElbowVisualizer(km, k=(2,10))
    visualizer.fit(data)
    return visualizer.elbow_value_

def silhouette(nonconverted_data):
    s_score = {} 
    for i in range(2,11):
        kmeans = KMeans(n_clusters=i, random_state=0).fit(nonconverted_data)
        classes = kmeans.predict(nonconverted_data)
        s_score[i] = (silhouette_score(nonconverted_data, classes))
    df = pd.DataFrame()
    df['k'] = s_score.keys()
    df['Silhouette Score'] = s_score.values()
    c1, c2 = st.columns(2)
    fig = px.line(df, x='k', y='Silhouette Score', markers=True)
    c1.plotly_chart(fig)

def aic_bic(x):
    aic_score = {} 
    bic_score = {}
    for i in range(1,11):
        gmm = GaussianMixture(n_components=i, random_state=0).fit(x)
        aic_score[i] = gmm.aic(x)
        bic_score[i] = gmm.bic(x)
    df = pd.DataFrame()
    df['k'] = aic_score.keys()
    df['aic'] = aic_score.values()
    df['bic'] = bic_score.values()
    c1, c2 = st.columns(2)
    fig1 = px.line(df, x='k', y='aic', markers=True)
    fig2 = px.line(df, x='k', y='bic', markers=True)
    c1.plotly_chart(fig1,use_container_width=True)
    c2.plotly_chart(fig2,use_container_width=True)

@st.cache_data()
def convert_df(df):
    return df.to_csv().encode('utf-8')


def df_font(data, background_color, font_color):
    return data.style.set_properties(**{'background-color': background_color,'color': font_color})

