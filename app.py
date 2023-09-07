import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
from segment import *

#Application title--------------------------------------------------------------------------------------------------
st.sidebar.header('Similarity Based Segmentation')

#Upload data--------------------------------------------------------------------------------------------------------
uploaded_file = st.sidebar.file_uploader('Please upload csv file:', type=['csv', 'xlsx', 'txt'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.to_csv('user_data.csv', index=False)

#Select all features or selected features---------------------------------------------------------------------------
is_all = st.sidebar.checkbox('All features')
if is_all:
    selected_columns = col
else:
    selected_columns = st.sidebar.multiselect('Please select attributes:',col)

#Select top similarity----------------------------------------------------------------------------------------------
# top_similarity = st.sidebar.number_input('Similarity based on top N:',min_value=1, max_value=10)
top_similarity = 5

#Extract the similar segment----------------------------------------------------------------------------------------
if st.sidebar.button('Extract Segment'):
    st.title('Users Segmentation Based on Similarity')
    st.write('''---''')

#Load data----------------------------------------------------------------------------------------------------------
    data = load_data()


#Split data into converted & non-converted--------------------------------------------------------------------------
    converted, nonconverted = split_conversion_nonconversion(data)


#Find optimal number of cluster automatically-----------------------------------------------------------------------
    optimal_k = elbow_method(nonconverted[selected_columns])

#Predict clusters---------------------------------------------------------------------------------------------------
    kmeanModel = KMeans(n_clusters=optimal_k)
    nonconverted['k'] = kmeanModel.fit_predict(nonconverted[selected_columns])

#Form a aggregated mean cluster on non-converted--------------------------------------------------------------------
    centroid_nonconverted = nonconverted.groupby('k',as_index=False)[col].agg('mean')
    centroid_nonconverted = pd.merge(centroid_nonconverted,
                                        nonconverted['k'].value_counts().to_frame().reset_index().rename(columns={'index':'k','k':'Count'}),on='k')

#Form a aggregated mean cluster on converted------------------------------------------------------------------------
    centroid_converted = converted[selected_columns].mean().to_frame().T

#Compute euclidean distance between non-converted & converted mean cluster------------------------------------------
    score = euclidean_distances(X=centroid_nonconverted[selected_columns], 
                                Y=centroid_converted)
    centroid_nonconverted['distance'] = score


#Results-------------------------------------------------------------------------------------------------------
#Non-converted cluster-----------------------------------------------------------------------------------------
    st.subheader('Non-converted cluster:')
    st.dataframe(df_font(centroid_nonconverted[selected_columns+['k','distance']], 
                         background_color='#8DB6CD',
                         font_color='white'), 
                         use_container_width=True)
    
#Most similar cluster------------------------------------------------------------------------------------------
    st.subheader('Most similare cluster:')
    c1, c2 = st.columns(2)
    c1.write('***Converted:***')
    c1.dataframe(df_font(centroid_converted,
                         background_color='#8DB6CD',
                         font_color='white'), 
                         use_container_width=False)

    c2.write('***Non-converted:***')
    c2.dataframe(df_font(centroid_nonconverted[selected_columns+['k','distance']][centroid_nonconverted['distance']==centroid_nonconverted['distance'].min()],
                         background_color='#8DB6CD',
                         font_color='white'), 
                         use_container_width=False)
    
    segment = centroid_nonconverted[selected_columns+['k','distance']][centroid_nonconverted['distance']==centroid_nonconverted['distance'].min()].k.values[0]
    st.write(f'***Segment number: {int(segment)}***')

#Extract the most similare segment of non-converted data---------------------------------------------------------
    segment_data = nonconverted[nonconverted['k']==segment].reset_index(drop=True)

#Converted data--------------------------------------------------------------------------------------------------
    st.subheader('Converted users data:')
    st.write(f'***Total converted data are {converted.shape[0]}***')
    st.dataframe(df_font(converted.reset_index(drop=True),
                        background_color='#8DB6CD',
                        font_color='white'), 
                        use_container_width=True)
    
#Top categories of converted data--------------------------------------------------------------------------------
    top_N = top_n()
    device = top_N[top_N['Category']=='Device'].iloc[:,1:3].reset_index(drop=True).rename(columns={'Subcategory':'Device'})
    browser = top_N[top_N['Category']=='Browser'].iloc[:,1:3].reset_index(drop=True).rename(columns={'Subcategory':'Browser'})
    region = top_N[top_N['Category']=='Region'].iloc[:,1:3].reset_index(drop=True).rename(columns={'Subcategory':'Region'})
    city = top_N[top_N['Category']=='City'].iloc[:,1:3].reset_index(drop=True).rename(columns={'Subcategory':'City'})
    landing_page = top_N[top_N['Category']=='Landing Pages'].iloc[:,1:3].reset_index(drop=True).rename(columns={'Subcategory':'Landing Pages'})

#Filtering the most similare segment of non-converted data based on characteristics-------------------------------
    similarity_based_device = segment_data.loc[segment_data.device_category.isin(device.head(top_similarity).Device)].reset_index(drop=True)
    similarity_based_browser = segment_data.loc[segment_data.web_browser.isin(browser.head(top_similarity).Browser)].reset_index(drop=True)
    similarity_based_region = segment_data.loc[segment_data.region.isin(region.head(top_similarity).Region)].reset_index(drop=True)
    similarity_based_city = segment_data.loc[segment_data.city.isin(city.head(top_similarity).City)].reset_index(drop=True)
    similarity_landing_page = segment_data.loc[segment_data.landing_page.isin(landing_page['Landing Pages'].head(top_similarity))].reset_index(drop=True)

    similar_segment = pd.concat([similarity_based_device, 
                                        similarity_based_browser, 
                                        similarity_based_region, 
                                        similarity_based_city,
                                        similarity_landing_page], axis=0,
                                        ignore_index=True)
    similar_segment = similar_segment.drop_duplicates('user_pseudo_id')

    st.subheader('Non-converted users data:')
    st.write(f'***Total most similar non-converted data similare to the converted are {similar_segment.shape[0]}***')
    st.dataframe(df_font(similar_segment,
                        background_color='#8DB6CD',
                        font_color='white'),
                        use_container_width=True)
    
    csv = convert_df(similar_segment)
    st.download_button(label="Download",
                        data=csv,
                        file_name=f'UserSegment{segment}.csv',
                        mime='text/csv')
