import pandas as pd
import streamlit as st

# Load the CSV file into a DataFrame
df = pd.read_csv('notebooks/prediction_results.csv')  # Replace 'your_file.csv' with the actual file path

# Streamlit App
st.sidebar.title('Location Recommendation based on App')

# App selection on the sidebar
selected_app = st.sidebar.selectbox('Select an App:', df['Recommended App'].unique())

# Filter dataframe based on selected app
filtered_df = df[df['Recommended App'] == selected_app]

# Group by location and calculate usage count
location_counts = filtered_df['Location'].value_counts().reset_index()
location_counts.columns = ['Location', 'Usage Count']

# Display recommended locations
if not location_counts.empty:
    st.write(f"Recommended locations for app '{selected_app}':")
    st.dataframe(location_counts)
else:
    st.write(f"No recommended locations found for app '{selected_app}'")
