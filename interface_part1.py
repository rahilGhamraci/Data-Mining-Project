import streamlit as st
import geopandas as gpd
import pandas as pd
import fiona

# Define functions for each page
def home_page():
    st.title("Home Page")
    st.write("Welcome to the Home Page of the Climate and Soil Algerian Data Study!")
    st.write("""
    This project focuses on exploratory data analysis (EDA) and data preprocessing using the climate dataset of Algeria. 
    Real-world data often comes with challenges like noise, missing values, and diverse sources. In this project, we aim to familiarize ourselves with the dataset, 
    clean and preprocess it, and identify key insights. The application allows for efficient data handling, from data cleaning and integration to normalization, 
    to improve the performance of mining algorithms. Let's dive into the data and uncover valuable information for better decision-making and analysis.
    """)


import pandas as pd
import geopandas as gpd
import streamlit as st

def load_csv_with_progress(file_path):
    """
    Load a CSV file with a progress bar.
    Assumes geometry is stored as WKT in the 'geometry' column.
    """
    # Determine the total number of lines in the CSV for progress tracking
    total_lines = sum(1 for _ in open(file_path)) - 1  # Exclude header line
    features = []
    progress_bar = st.progress(0)
    
    with open(file_path, "r") as f:
        header = f.readline().strip().split(",")
        for idx, line in enumerate(f):
            features.append(line.strip().split(","))
            progress = int((idx + 1) / total_lines * 100)
            progress_bar.progress(progress)
    
    progress_bar.empty()
    
    # Create a DataFrame and convert geometry column back to geometry objects
    df = pd.DataFrame(features, columns=header)
 
    return df

def load_data_with_progress(file_path):
    """
    Load a GeoPackage file with a progress bar.
    """
    with fiona.open(file_path, "r") as src:
        total_features = len(src)
        features = []
        progress_bar = st.progress(0)
        
        for idx, feature in enumerate(src):
            features.append(feature)
            progress = int((idx + 1) / total_features * 100)
            progress_bar.progress(progress)
        
        progress_bar.empty()
        
        gdf = gpd.GeoDataFrame.from_features(features, crs=src.crs)
        gdf['geometry'] = gdf['geometry'].apply(lambda x: x.wkt if x is not None else None)
        
        return gdf

# Streamlit app
def data_loading_page():
    st.title("Data Loading Page")
    
    data_choice = st.selectbox("Choose the type of data to load", ["", "Climate", "Soil"])
    
    if data_choice == "":
        st.error("Please select a data type.")
    else:
        if data_choice == "Climate" and "climate_gdf" not in st.session_state:
            st.write("Loading Climate Data...")
            climate_gdf = load_csv_with_progress("final_climate_data.csv")
            st.session_state.climate_gdf = climate_gdf
            st.dataframe(climate_gdf)
            st.success("Climate Data loaded successfully!")

        elif data_choice == "Soil" and "soil_gdf" not in st.session_state:
            st.write("Loading Soil Data...")
            soil_gdf = load_data_with_progress("soil_data.gpkg")
            st.session_state.soil_gdf = soil_gdf
            st.dataframe(soil_gdf)
            st.success("Soil Data loaded successfully!")

        else:
            st.write(f"{data_choice} data is already loaded.")
            if data_choice == "Climate":
                st.dataframe(st.session_state.climate_gdf)
            elif data_choice == "Soil":
                st.dataframe(st.session_state.soil_gdf)

        
def description(df):
    colonnes_description = []
    for d in df:
        colonnes_description.append([d, df[d].count(), str(df.dtypes[d])])
        df.DataFrame(colonnes_description, columns = ["Name","Values not null","Type"])
    return 

def data_description_page():
    st.title("Data Description")
    description_choice = st.selectbox("Choose the type of data to describe", ["Climate Global Description", "Climate Columns Description", "Soil Global Description", "Soil Columns Description"])

def data_modification_page():
    st.title("Data Modification Page")
    st.write("This is the Data Modification Page.")

def central_tendencies_page():
    st.title("Central Tendencies Page")
    st.write("This is the Central Tendencies Page")

def dispersion_measures_page():
    st.title("Dispersion Measures Page")
    st.write("This is the Dispersion Measures Page")

def missing_unique_values_page():
    st.title("Missing Unique Values Page")
    st.write("Missing Unique Values Page")

def boxplots_histograms_page():
    st.title("Boxplots Histograms Page")
    st.write("Boxplots and Histograms")

def scatter_plots_page():
    st.title("Scatter Plots Page")
    st.write("Scatter Plots")

def handling_outliers_page():
    st.title("Handling Outliers Page")
    st.write("Handling Outliers")

def handling_missing_values_page():
    st.title("Handling Missing Values Page")
    st.write("Missing Values")

def data_reduction_page():
    st.title("Data Reduction Page")
    st.write("Season Agregation")

def data_integration_page():
    st.title("Data Integration Page")
    st.write("Data Integration")

def redundancies_removal_page():
    st.title("Redundancies Removal Page")
    st.write("Removing Redundancies.")

def normalization_page():
    st.title("Normalization Page")
    st.write("Normalization of Data.")



# Sidebar menu
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Data Manipulation", "Characteristics Analysis", "Preprocessing"])

# Add submenus for each main menu option
if menu == "Home":
    home_page()

elif menu == "Data Manipulation":
    submenu = st.sidebar.radio("Data Manipulation Submenu", ["Import Dataset", "Global Description", "Update/Delete"])
    if submenu == "Import Dataset":
        data_loading_page()
    elif submenu == "Global Description":
        data_description_page()
    elif submenu == "Update/Delete":
        data_modification_page()

elif menu == "Characteristics Analysis":
    submenu = st.sidebar.radio("Characteristics Analysis Submenu", [
        "Central Tendencies", "Dispersion Measures", 
        "Missing and Unique Values", "Boxplots and Histograms", 
        "Scatter Plots"
    ])
    if submenu == "Central Tendencies":
        central_tendencies_page()
    elif submenu == "Dispersion Measures":
        dispersion_measures_page()
    elif submenu == "Missing and Unique Values":
        missing_unique_values_page()
    elif submenu == "Boxplots and Histograms":
        boxplots_histograms_page()
    elif submenu == "Scatter Plots":
        scatter_plots_page()

elif menu == "Preprocessing":
    submenu = st.sidebar.radio("Preprocessing Submenu", [
        "Handling Outliers", "Handling Missing Values", 
        "Data Reduction: Season Agregation", "Data Integration", 
        "Redundancies Removal", "Normalization"
    ])
    if submenu == "Handling Outliers":
        handling_outliers_page()
    elif submenu == "Handling Missing Values":
        handling_missing_values_page()
    elif submenu == "Data Reduction: Season Agregation":
        data_reduction_page()
    elif submenu == "Data Integration":
        data_integration_page()
    elif submenu == "Redundancies Removal":
        redundancies_removal_page()
    elif submenu == "Normalization":
        normalization_page()
