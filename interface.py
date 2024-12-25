import streamlit as st
import geopandas as gpd
import pandas as pd
import fiona
import matplotlib.pyplot as plt
from PIL import Image
import os
import pickle
import numpy as np



    
def home_page():
    

   
    st.title("Home Page")
    st.write("Welcome to the Home Page of the Climate and Soil Algerian Data Study!")
    st.write("""
    This project focuses on exploratory data analysis (EDA) and data preprocessing using the climate dataset of Algeria. 
    Real-world data often comes with challenges like noise, missing values, and diverse sources. In this project, we aim to familiarize ourselves with the dataset, 
    clean and preprocess it, and identify key insights. The application allows for efficient data handling, from data cleaning and integration to normalization, 
    to improve the performance of mining algorithms. Let's dive into the data and uncover valuable information for better decision-making and analysis.
    """)
    st.image(Image.open("Images/climate_soil_data_map.png"), use_container_width=True)
#......................................................................................................................


def load_csv_with_progress(file_path):
   
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

def load_climate_csv_with_progress(file_path, max_rows=5000):
    total_lines = sum(1 for _ in open(file_path)) - 1  
    features = []
    progress_bar = st.progress(0)

    with open(file_path, "r") as f:
        header = f.readline().strip().split(",")  
        for idx, line in enumerate(f):
            if idx >= max_rows: 
                break
            features.append(line.strip().split(","))
            progress = int((idx + 1) / min(total_lines, max_rows) * 100)
            progress_bar.progress(progress)
    
    progress_bar.empty()

    df = pd.DataFrame(features, columns=header)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df



# Streamlit app
def data_loading_page():
    st.title("Data Loading Page")
    
    data_choice = st.selectbox("Choose the type of data to load", ["", "Climate", "Soil"])
    
    if data_choice == "":
        st.error("Please select a data type.")
    else:
        if data_choice == "Climate" and "climate_gdf" not in st.session_state:
            st.write("Loading Climate Data...")
            climate_gdf = load_climate_csv_with_progress("final_climate_data.csv")
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

#......................................................................................................................      


def description(df):
    """
    Generate a DataFrame containing a description of the input DataFrame.
    """
    colonnes_description = []

    for col in df.columns:
        colonnes_description.append([
            col,  
            df[col].count(), 
            str(df[col].dtype)  
        ])

    desc_df = pd.DataFrame(colonnes_description, columns=["Name", "Values not null", "Type"])
    return desc_df

def save_description(description_df, file_name):
    """
    Save the description DataFrame to a file using pickle.
    """
    with open(file_name, 'wb') as f:
        pickle.dump(description_df, f)

def load_description(file_name):
    """
    Load the description DataFrame from a file.
    """
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    else:
        return None

def data_description_page():

    st.title("Data Description")

    description_choice = st.selectbox(
        "Choose the type of data to describe", 
        ["", "Climate Global Description", "Soil Global Description"]
    )

    # File paths for saved descriptions
    climate_desc_file = "climate_description.pkl"
    soil_desc_file = "soil_description.pkl"

    if description_choice == "Climate Global Description":

        climate_desc = load_description(climate_desc_file)
        
        if climate_desc is None:
            climate_desc = description(st.session_state.climate_gdf)
            save_description(climate_desc, climate_desc_file)
            st.dataframe(climate_desc)
            
        else:
            st.dataframe(climate_desc)

    elif description_choice == "Soil Global Description":
        soil_desc = load_description(soil_desc_file)
        
        if soil_desc is None:
           soil_desc = description(st.session_state.soil_gdf)
           save_description(soil_desc, soil_desc_file)
           st.dataframe(soil_desc)
        else:
            st.dataframe(soil_desc)
#......................................................................................................................

def data_modification_page():
    st.title("Data Modification Page")
    st.write("Select an attribute, operation, and values to modify the dataset.")

    # Attribute categories
    soil_attributes = [
        'sand % topsoil', 'sand % subsoil', 'silt % topsoil', 'silt% subsoil',
        'clay % topsoil', 'clay % subsoil', 'pH water topsoil',
        'pH water subsoil', 'OC % topsoil', 'OC % subsoil', 'N % topsoil',
        'N % subsoil', 'BS % topsoil', 'BS % subsoil', 'CEC topsoil',
        'CEC subsoil', 'CEC clay topsoil', 'CEC Clay subsoil',
        'CaCO3 % topsoil', 'CaCO3 % subsoil', 'BD topsoil', 'BD subsoil',
        'C/N topsoil', 'C/N subsoil'
    ]

    climate_attributes = [
        'PSurf', 'Qair', 'Rainf', 'Snowf', 'Tair', 'Wind', 'lon', 'lat'
    ]

    # Dropdowns for attribute selection, operation, and value input
    attribute_choice = st.selectbox("Choose an attribute to modify", soil_attributes + climate_attributes)
    operation = st.selectbox("Choose an operation", ["", "Delete", "Update"])
    value = st.text_input("Enter the value to modify")
    new_value = st.text_input("Enter the new value (leave blank for delete)") if operation == "Update" else None

    # Determine the target dataset
    if attribute_choice in soil_attributes:
        target_gdf_key = "soil_gdf"
        target_dataset = st.session_state.get("soil_gdf")
    elif attribute_choice in climate_attributes:
        target_gdf_key = "climate_gdf"
        target_dataset = st.session_state.get("climate_gdf")
    else:
        st.error("Invalid attribute selected.")
        return

    if target_dataset is None:
        st.warning(f"Please load the dataset for {target_gdf_key} first.")
        return

    # Perform the operation
    if operation == "Delete" or operation == "Update":
        if attribute_choice in target_dataset.columns:
            try:
                # Convert value and new_value to appropriate types
                value = float(value) if value.replace('.', '', 1).isdigit() else value
                new_value = float(new_value) if new_value and new_value.replace('.', '', 1).isdigit() else new_value

                if operation == "Update" and new_value is not None:
                    # Ensure consistent data type
                    target_dataset[attribute_choice] = target_dataset[attribute_choice].astype(type(new_value))
                    target_dataset.loc[target_dataset[attribute_choice] == value, attribute_choice] = new_value
                    # Reinforce data type consistency
                    target_dataset[attribute_choice] = target_dataset[attribute_choice].astype(type(new_value))
                    st.success(f"Updated '{attribute_choice}' where value is '{value}' to '{new_value}'.")
                elif operation == "Delete":
                    target_dataset = target_dataset[target_dataset[attribute_choice] != value]
                    st.success(f"Deleted rows where '{attribute_choice}' is '{value}'.")
            except Exception as e:
                st.error(f"Failed to perform the operation: {e}")
        else:
            st.error(f"Attribute '{attribute_choice}' does not exist in the dataset.")
    else:
        if operation:
            st.warning("Please choose a valid operation.")

    # Save modified dataset back to session state
    st.session_state[target_gdf_key] = target_dataset

    st.write(f"Modified {target_gdf_key} dataset:")
    st.dataframe(target_dataset)

#......................................................................................................................
def load_csv(file_path):
    return pd.read_csv(file_path, sep=';')

def data_summary_page():
    st.title("Data Summary Page: Central tendecies, Missing / Unqiue Values, Outliers")
    data_choice = st.selectbox("Choose the type of data ", ["", "Climate", "Soil"])
    
    if data_choice == "":
        st.error("Please select a data type.")
    else:
        if data_choice == "Climate" and "climate_attribute_summary" not in st.session_state:
            
            climate_attribute_summary = load_csv("climate_attributes_summary.csv")
            st.session_state.climate_attribute_summary = climate_attribute_summary
            st.dataframe(climate_attribute_summary)

        elif data_choice == "Soil" and "soil_attribute_summary" not in st.session_state:
            
            soil_attribute_summary = load_csv("soil_attributes_summary.csv")
            st.session_state.soil_attribute_summary = soil_attribute_summary
            st.dataframe(soil_attribute_summary)

        else:
            st.write(f"{data_choice} summray is already loaded.")
            if data_choice == "Climate":
                st.dataframe(st.session_state.climate_attribute_summary)
            elif data_choice == "Soil":
                st.dataframe(st.session_state.soil_attribute_summary)

#......................................................................................................................

def boxplots_histograms_page():
    
    st.title("Boxplots and Histograms Page")
    st.write("Select an attribute to view its bar chart and box plot.")

    # Attribute lists
    soil_attributes = [
        'sand % topsoil', 'sand % subsoil', 'silt % topsoil', 'silt% subsoil',
        'clay % topsoil', 'clay % subsoil', 'pH water topsoil',
        'pH water subsoil', 'OC % topsoil', 'OC % subsoil', 'N % topsoil',
        'N % subsoil', 'BS % topsoil', 'BS % subsoil', 'CEC topsoil',
        'CEC subsoil', 'CEC clay topsoil', 'CEC Clay subsoil',
        'CaCO3 % topsoil', 'CaCO3 % subsoil', 'BD topsoil', 'BD subsoil',
        'C/N topsoil', 'C/N subsoil'
    ]

    climate_attributes = [
        'PSurf', 'Qair', 'Rainf', 'Snowf', 'Tair', 'Wind', 'lon', 'lat'
    ]

    
    all_attributes = soil_attributes + climate_attributes

    
    attribute_choice = st.selectbox("Choose an attribute to visualize", all_attributes)

    if attribute_choice:
        
        safe_attribute_name = attribute_choice.replace("/", "_") 

        # File paths
        boxplot_path = os.path.join("boxplots", f"{safe_attribute_name}_box_plot.png")
        barchart_path = os.path.join("barcharts", f"{safe_attribute_name}_bar_chart.png")

        # Display the images
        col1, col2 = st.columns(2)  # Create two columns for side-by-side display

        try:
            with col1:
                st.image(Image.open(barchart_path), caption=f"Bar Chart: {attribute_choice}", use_container_width=True)
            with col2:
                st.image(Image.open(boxplot_path), caption=f"Box Plot: {attribute_choice}", use_container_width=True)
        except FileNotFoundError as e:
            st.error(f"Could not find the image for {attribute_choice}. Ensure the files are located in the correct folders.")

    else:
        st.info("Please select an attribute to view its visualizations.")


#......................................................................................................................
# Scatter plot function
def scatter_plot(attribute_name_x, attribute_name_y, attribute_values_x, attribute_values_y):
    """
    Generate and display a scatter plot for the given attributes and their values.
    """
    plt.figure(figsize=(8, 6))
    plt.title("Scatter Plot")
    plt.xlabel(attribute_name_x)
    plt.ylabel(attribute_name_y)
    plt.scatter(attribute_values_x, attribute_values_y, alpha=0.7)
    plt.grid(True)
    st.pyplot(plt)  

# Scatter Plots Page
def scatter_plots_page():
   
    st.title("Scatter Plots Page")
    st.write("Select two attributes to visualize their scatter plot.")

    # Attribute lists
    soil_attributes = [
        'sand % topsoil', 'sand % subsoil', 'silt % topsoil', 'silt% subsoil',
        'clay % topsoil', 'clay % subsoil', 'pH water topsoil',
        'pH water subsoil', 'OC % topsoil', 'OC % subsoil', 'N % topsoil',
        'N % subsoil', 'BS % topsoil', 'BS % subsoil', 'CEC topsoil',
        'CEC subsoil', 'CEC clay topsoil', 'CEC Clay subsoil',
        'CaCO3 % topsoil', 'CaCO3 % subsoil', 'BD topsoil', 'BD subsoil',
        'C/N topsoil', 'C/N subsoil'
    ]

    climate_attributes = [
        'PSurf', 'Qair', 'Rainf', 'Snowf', 'Tair', 'Wind', 'lon', 'lat'
    ]

    # Combine attribute lists
    all_attributes = soil_attributes + climate_attributes

    # Dropdowns for selecting attributes
    attribute_choice_x = st.selectbox("Select X-axis attribute", all_attributes)
    attribute_choice_y = st.selectbox("Select Y-axis attribute", all_attributes)

    # Ensure both attributes are selected
    if attribute_choice_x and attribute_choice_y:
        # Check if attributes belong to soil or climate data
        if attribute_choice_x in soil_attributes and attribute_choice_y in soil_attributes:
            df = st.session_state.get("soil_gdf", None)  # Soil dataset
        elif attribute_choice_x in climate_attributes and attribute_choice_y in climate_attributes:
            df = st.session_state.get("climate_gdf", None)  # Climate dataset
        else:
            df = None
            st.error("Attributes must belong to the same dataset (either Soil or Climate).")

        # Generate scatter plot if dataset is available
        if df is not None:
            try:
                attribute_values_x = df[attribute_choice_x]
                attribute_values_y = df[attribute_choice_y]
                scatter_plot(attribute_choice_x, attribute_choice_y, attribute_values_x, attribute_values_y)
            except KeyError:
                st.error(f"One or both attributes ({attribute_choice_x}, {attribute_choice_y}) are not available in the dataset.")
    else:
        st.info("Please select both X-axis and Y-axis attributes to view the scatter plot.")

#......................................................................................................................

# List of attributes for soil data
soil_attributes = ['sand % topsoil', 'sand % subsoil', 'silt % topsoil', 'silt% subsoil',
    'clay % topsoil', 'clay % subsoil', 'pH water topsoil',
    'pH water subsoil', 'OC % topsoil', 'OC % subsoil', 'N % topsoil',
    'N % subsoil', 'BS % topsoil', 'BS % subsoil', 'CEC topsoil',
    'CEC subsoil', 'CEC clay topsoil', 'CEC Clay subsoil',
    'CaCO3 % topsoil', 'CaCO3 % subsoil', 'BD topsoil', 'BD subsoil',
    'C/N topsoil', 'C/N subsoil']

soil_attributes_with_outliers = ['OC % topsoil', 'N % topsoil', 'N % subsoil', 'CEC topsoil', 
                                'CEC subsoil', 'CaCO3 % topsoil', 'CaCO3 % subsoil']

# List of attributes for climate data
climate_attributes = ['PSurf', 'Qair', 'Rainf', 'Snowf', 'Tair', 'Wind', 'lon', 'lat']

climate_attributes_with_outliers = ['PSurf', 'Qair', 'Rainf', 'Snowf', 'Tair', 'Wind']

# Function to highlight outliers in bold
def highlight_bold(val, changed_values):
    return f'**{val}**' if val in changed_values else val

# Function to handle outliers for soil data
def handle_soil_outliers(df, columns, strategy='mean', iqr_multiplier=1.5):
    df_cleaned = df.copy()
    changed_values = []

    for column_name in columns:
        if df_cleaned[column_name].dtype not in ['float64', 'int64']:
            continue

        column_data = df_cleaned[column_name].fillna(df_cleaned[column_name].median())  
        
        q1 = column_data.quantile(0.25)
        q3 = column_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        outliers = (column_data < lower_bound) | (column_data > upper_bound)
        num_outliers = outliers.sum()
        print(f"{column_name}: {num_outliers} outliers detected.")

        if strategy == 'mean':
            replacement_value = np.float64(column_data.mean())
        elif strategy == 'median':
            replacement_value = np.float64(column_data.median())
        elif strategy == 'mode':
            replacement_value = np.float64(column_data.mode()[0] )
        elif strategy == 'quantile':
            df_cleaned.loc[column_data < lower_bound, column_name] = q1
            df_cleaned.loc[column_data > upper_bound, column_name] = q3
            continue  
        elif strategy == 'cap':
            df_cleaned.loc[column_data < lower_bound, column_name] = lower_bound
            df_cleaned.loc[column_data > upper_bound, column_name] = upper_bound
            continue
        elif strategy == 'random':
            random_values = np.random.uniform(q1, q3, size=num_outliers).astype('float64')
            df_cleaned.loc[outliers & ~df_cleaned[column_name].isna(), column_name] = random_values
            remaining_outliers = ((df_cleaned[column_name] < lower_bound) | (df_cleaned[column_name] > upper_bound)).sum()
            print(f"{column_name}: Outliers after handling: {remaining_outliers}")
            continue
        else:
            raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', 'quantile', 'cap', or 'random'.")

        # Collect changed values for bold highlighting
        changed_values.extend(df_cleaned[column_name][outliers].values)

        df_cleaned.loc[outliers & ~df_cleaned[column_name].isna(), column_name] = replacement_value
        remaining_outliers = ((df_cleaned[column_name] < lower_bound) | (df_cleaned[column_name] > upper_bound)).sum()
        print(f"{column_name}: Outliers after handling: {remaining_outliers}")

    return df_cleaned, changed_values

# Function to handle outliers for climate data
def handle_climate_outliers(df, columns, strategy='mean', iqr_multiplier=1.5):
    df_cleaned = df.copy()
    changed_values = []

    for column_name in columns:
            
            if df_cleaned[column_name].dtype not in ['float64', 'float32', 'int64']:
                continue

            # Calculate IQR and bounds
            column_data = df_cleaned[column_name]
            q1 = column_data.quantile(0.25)
            q3 = column_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr

            # Identify outliers (align indices with the original DataFrame)
            outliers = (column_data < lower_bound) | (column_data > upper_bound)
            num_outliers = outliers.sum()
            print(f"{column_name}: {num_outliers} outliers detected.")

            # Handle outliers based on the strategy
            if strategy == 'mean':
                replacement_value = np.float32(column_data.mean())
                df_cleaned.loc[outliers, column_name] = replacement_value
            elif strategy == 'median':
                replacement_value = np.float32(column_data.median())
                df_cleaned.loc[outliers, column_name] = replacement_value
            elif strategy == 'mode':
                replacement_value = np.float32(column_data.mode()[0])
                df_cleaned.loc[outliers, column_name] = replacement_value
            elif strategy == 'quantile':
                df_cleaned.loc[column_data < lower_bound, column_name] = np.float32(q1)
                df_cleaned.loc[column_data > upper_bound, column_name] = np.float32(q3)
            elif strategy == 'cap':
                df_cleaned.loc[column_data < lower_bound, column_name] = lower_bound
                df_cleaned.loc[column_data > upper_bound, column_name] = upper_bound
            elif strategy == 'random':
                random_values = np.random.uniform(q1, q3, size=num_outliers).astype('float32')
                df_cleaned.loc[outliers, column_name] = random_values
            else:
                raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', 'quantile', 'cap', or 'random'.")

            # Collect changed values for bold highlighting
            changed_values.extend(df_cleaned.loc[outliers, column_name].values)

            remaining_outliers = ((df_cleaned[column_name] < lower_bound) | (df_cleaned[column_name] > upper_bound)).sum()
            print(f"{column_name}: Outliers after handling: {remaining_outliers}")

    return df_cleaned, changed_values



# Main page to handle outliers based on selected data type (soil or climate)
def handling_outliers_page():
    st.title("Handling Outliers Page")
    st.write("Handling Outliers")

    # User selects which data to process
    data_type = st.selectbox("Select Data Type", ("Soil", "Climate"))

    if data_type == "Soil":
        # Get the soil DataFrame from session state
        df = st.session_state.get("soil_gdf", None)
        if df is not None:
            # User selects the attribute to handle outliers for
            attribute = st.selectbox("Select Attribute", soil_attributes + soil_attributes_with_outliers)
            
            if attribute in soil_attributes_with_outliers:
                # Handle outliers using the corresponding function for soil data
                strategy = st.selectbox("Select Handling Strategy", ("mean", "median", "mode", "quantile", "cap", "random"))
                cleaned_df, changed_values = handle_soil_outliers(df, [attribute], strategy=strategy)
                st.write(f"Outliers handled for {attribute} with strategy: {strategy}")

                # Apply highlight_bold function
                cleaned_df = cleaned_df.applymap(lambda x: highlight_bold(x, changed_values))
                st.write(cleaned_df[attribute])
            else:
                st.write(f"No outlier handling required for {attribute}.")
        else:
            st.write("Soil data is not available.")

    elif data_type == "Climate":
        # Get the climate DataFrame from session state
        df = st.session_state.get("climate_gdf", None)
        if df is not None:
            # User selects the attribute to handle outliers for
            attribute = st.selectbox("Select Attribute", climate_attributes + climate_attributes_with_outliers)
            
            if attribute in climate_attributes_with_outliers:
                # Handle outliers using the corresponding function for climate data
                strategy = st.selectbox("Select Handling Strategy", ("mean", "median", "mode", "quantile", "cap", "random"))
                cleaned_df, changed_values = handle_climate_outliers(df, [attribute], strategy=strategy)
                st.write(f"Outliers handled for {attribute} with strategy: {strategy}")

                # Apply highlight_bold function
                cleaned_df = cleaned_df.applymap(lambda x: highlight_bold(x, changed_values))
                st.write(cleaned_df[attribute])
            else:
                st.write(f"No outlier handling required for {attribute}.")
        else:
            st.write("Climate data is not available.")


#......................................................................................................................

# List of climate attributes
climate_attributes = ['PSurf', 'Qair', 'Rainf', 'Snowf', 'Tair', 'Wind', 'lon', 'lat']

# Function for handling missing values
def handling_missing_values(df, attributes, choice):
    updated_df = df.copy()

    # Ensure attributes are numeric
    for attribute in attributes:
        if attribute in updated_df.columns:
            updated_df[attribute] = pd.to_numeric(updated_df[attribute], errors='coerce')

    match choice:
        case 'Ignore the Nan values':
            updated_df.dropna(axis=0, inplace=True)
            return updated_df

        case 'Replace by Mean':
            for attribute in attributes:
                if attribute in updated_df.columns:
                    updated_df[attribute] = updated_df[attribute].fillna(updated_df[attribute].mean())
            return updated_df

        case 'Replace by Median':
            for attribute in attributes:
                if attribute in updated_df.columns:
                    updated_df[attribute] = updated_df[attribute].fillna(updated_df[attribute].median())
            return updated_df

        case 'Replace using Bayes Formula':
            for attribute in attributes:
                if attribute in updated_df.columns:
                    overall_mean = updated_df[attribute].mean()
                    overall_std = updated_df[attribute].std()

                    nan_indices = updated_df[updated_df[attribute].isna()].index
                    updated_df.loc[nan_indices, attribute] = np.random.normal(loc=overall_mean, scale=overall_std, size=len(nan_indices))

            return updated_df

# Streamlit page to handle missing values
def handling_missing_values_page():
    st.title("Handling Missing Values Page")

    # Get the climate DataFrame from session state
    df = st.session_state.get("climate_gdf", None)
    climate_attributes = ['PSurf', 'Qair', 'Rainf', 'Snowf', 'Tair', 'Wind', 'lon', 'lat']

    if df is not None:
        # User selects an attribute
        attribute = st.selectbox("Select an attribute to handle missing values:", climate_attributes)

        if attribute:
            # User selects which option to handle missing values
            choice = st.selectbox(
                "How would you like to handle missing values for the selected attribute?",
                ['Ignore the Nan values', 'Replace by Mean', 'Replace by Median', 'Replace using Bayes Formula']
            )

            # Apply missing values handling
            if choice:
                updated_df = handling_missing_values(df, [attribute], choice)
                st.write(f"Missing values handled for `{attribute}` using: {choice}")
                st.write(updated_df[attribute])
    else:
        st.write("Climate data is not available.")


#......................................................................................................................
def data_reduction_page():
    st.title("Data Reduction Page : Season Agregation")
    if "seasonal_climate_data" not in st.session_state:
        seasonal_climate_data = load_csv_with_progress("seasonal_climate_data.csv")
        st.session_state.seasonal_climate_data = seasonal_climate_data
        st.dataframe(seasonal_climate_data)
    else:
        st.dataframe(st.session_state.seasonal_climate_data)
#......................................................................................................................
def data_integration_page():
    st.title("Data Integration Page")
    if "final_merged_data" not in st.session_state:
        final_merged_data = load_csv_with_progress("final_merged_data.csv")
        st.session_state.final_merged_data = final_merged_data
        st.dataframe(final_merged_data)
    else:
        st.dataframe(st.session_state.final_merged_data)
#......................................................................................................................
def redundancies_removal_page():
    st.title("Redundancies Removal Page")
    data_choice = st.selectbox("Choose an option", ["", "Horizontal", "Correlation-Based Removal", "Low Variance Columns", "Customized Removal"])
    
    if data_choice == "":
        st.error("Please select a removal option.")
    else:
        if data_choice == "Horizontal" and "final_merged_data" not in st.session_state:
            
            final_merged_data = load_csv_with_progress("final_merged_data.csv")
            st.session_state.final_merged_data = final_merged_data
            st.dataframe(final_merged_data)

        elif data_choice == "Correlation-Based Removal" and "reduced_correlation_rm_df" not in st.session_state:
            
            reduced_correlation_rm_df = load_csv_with_progress("reduced_correlation_rm_df.csv")
            st.session_state.reduced_correlation_rm_df = reduced_correlation_rm_df
            st.dataframe(reduced_correlation_rm_df)

        elif data_choice == "Low Variance Columns" and "reduced_lvar_df" not in st.session_state:
            
            reduced_lvar_df = load_csv_with_progress("reduced_lvar_df.csv")
            st.session_state.reduced_lvar_df = reduced_lvar_df
            st.dataframe(reduced_lvar_df)

        elif data_choice == "Customized Removal" and "reduced_v_customized_df" not in st.session_state:
            
            reduced_v_customized_df = load_csv_with_progress("reduced_v_customized_df.csv")
            st.session_state.reduced_v_customized_df = reduced_v_customized_df
            st.dataframe(reduced_v_customized_df)

        else:
            if data_choice == "Horizontal":
                st.dataframe(st.session_state.final_merged_data)
            elif data_choice == "Correlation-Based Removal":
                st.dataframe(st.session_state.reduced_correlation_rm_df)
            elif data_choice == "Low Variance Columns":
                st.dataframe(st.session_state.reduced_lvar_df)
            elif data_choice == "Customized Removal":
                st.dataframe(st.session_state.reduced_v_customized_df)

#......................................................................................................................
def normalization_page():
    st.title("Normalization Page")
    data_choice = st.selectbox("Choose an option", ["", "min max normalization", "z_score normalization"])
    
    if data_choice == "":
        st.error("Please select a normalization option.")
    else:
        if data_choice == "min max normalization" and "min_max_normalized_data" not in st.session_state:
            
            min_max_normalized_data = load_csv_with_progress("min_max_normalized_data.csv")
            st.session_state.min_max_normalized_data = min_max_normalized_data
            st.dataframe(min_max_normalized_data)

        elif data_choice == "z_score normalization" and "z_score_normalized_data" not in st.session_state:
            
            z_score_normalized_data = load_csv_with_progress("z_score_normalized_data.csv")
            st.session_state.z_score_normalized_data = z_score_normalized_data
            st.dataframe(z_score_normalized_data)
        else:
            if data_choice == "min max normalization":
                st.dataframe(st.session_state.min_max_normalized_data)
            elif data_choice == "z_score normalization":
                st.dataframe(st.session_state.z_score_normalized_data)
#......................................................................................................................       

import nbimporter
from part2 import Node,load_tree,predict_w_base_learners,train,decision_tree_algorithm,predict,prepare_data

def print_tree(node, depth=0):
    st.write("  " * depth + str(node.data))
    for child in node.children:
        print_tree(child, depth + 1)

def models_page():
    st.title("Models Page")

    attributes = ['winter_PSurf', 'spring_PSurf', 'summer_PSurf', 'autumn_PSurf',
                  'winter_Tair', 'spring_Tair', 'summer_Tair', 'autumn_Tair',
                  'winter_Wind', 'spring_Wind', 'summer_Wind', 'autumn_Wind',
                  'lon', 'lat',
                  'sand % topsoil', 'silt % topsoil', 'clay % topsoil',
                  'OC % topsoil', 'OC % subsoil', 'N % topsoil', 'N % subsoil',
                  'CaCO3 % topsoil']

            
    
    choice = st.selectbox("Choose an method", ["", "Clustering", "Regression"])
    
    if choice == "":
        st.error("Please select a method.")
    else:
        if choice == "Clustering" :

            algo_choice = st.selectbox("Choose an algorithm", ["", "CLARANS", "DBSCAN"])

            if algo_choice == "CLARANS" :
                
                k = st.selectbox("Select the value of k:", ["2", "3", "4", "5", "6"])
                numlocal = st.selectbox("Select the value of numlocal:", ["6", "10", "15", "20"])
                maxneighbor = st.selectbox("Select the value of maxneighbor:", ["4", "6", "8", "10"])
                if k and numlocal and maxneighbor:
                    path = f"Clustering Results/Clarans/CLARANS Clusters (k={k}, numlocal={numlocal}, maxneighbor={maxneighbor}).png"
                    try:
                        st.image(Image.open(path), use_container_width=True)
                    except FileNotFoundError:
                        st.error(f"File not found: {path}")
                else:
                    st.warning("Please select all values for k, numlocal, and maxneighbor.")
             
            else:

                Eps = st.selectbox("Select the value of Eps:", ["0.5", "1", "2", "5"])
                MinPts = st.selectbox("Select the value of MinPts:", ["3", "5", "10", "15"])
               
                if Eps and MinPts: 
                    path = f"Clustering Results/DBSCAN/DBSCAN Clusters (Eps={Eps}, MinPts={MinPts}).png"
                    try:
                        st.image(Image.open(path), use_container_width=True)
                    except FileNotFoundError:
                        st.error(f"File not found: {path}")
                else:
                    st.warning("Please select values for Eps and MinPts.")
            
    

        elif choice == "Regression":
            #  Target and Algorithm Selection
            
            algo_choice = st.selectbox("Choose an algorithm", ["", "Decision Tree", "Random Forest"])

            if not algo_choice:
                st.warning("Please select a valid algorithm.")
            else:
      
                if algo_choice == "Decision Tree":

                    target_choice = st.selectbox("Choose a target variable", ["", "Qair winter", "Qair spring", "Qair summer", "Qair autumn"])
                    
                    target_map = {
                       "Qair winter": "decision_tree_model_winter.pkl",
                        "Qair spring": "decision_tree_model_spring.pkl",
                         "Qair summer": "decision_tree_model_summer.pkl",
                         "Qair autumn": "decision_tree_model_autumn.pkl"}
                    
                    target = target_map.get(target_choice, None)

                    if not target_choice:
                        st.warning("Please select a target")
                    else:
                        with st.spinner("Loading Decision Tree model..."):
                            root = load_tree(target)
                        
                    
                        st.success("Decision Tree model loaded successfully!")
                        print_tree(root)
                   

                        sample_instance = {}
                        st.write("Enter values for the following attributes:")

                        for attribute in attributes:
                            value = st.text_input(f"{attribute}:", "0")
                            sample_instance[attribute] = float(value)
                        
                        if sample_instance:
                            with st.spinner("Start Predecting..."):
                                prediction = predict(root, None, sample_instance)
                            st.write("Prediction for the entered instance:", prediction)
                       

         
                elif algo_choice == "Random Forest":

                    target_choice = st.selectbox("Choose a target variable", ["", "Qair winter", "Qair spring", "Qair summer", "Qair autumn"])
                    
                    target_map = {
                       "Qair winter": "random_forest_model_winter.pkl",
                        "Qair spring": "random_forest_model_spring.pkl",
                         "Qair summer": "random_forest_model_summer.pkl",
                         "Qair autumn": "random_forest_model_autumn.pkl"}
                    
                    target = target_map.get(target_choice, None)

                    if not target_choice:
                        st.warning("Please select a target")
                    else:
                        with st.spinner("Loading Random Forest model..."):
                            base_learners = load_tree(target)
                        
                    
                        st.success("Random Forest model loaded successfully!")

                        sample_instance = {}
                        st.write("Enter values for the following attributes:")

                        for attribute in attributes:
                            value = st.text_input(f"{attribute}:", "0")
                            sample_instance[attribute] = float(value)
                        
                        if sample_instance:
                            with st.spinner("Start Predecting..."):
                                all_predictions = []
                                for base_learner in base_learners:
                                    prediction = predict(base_learner, None, sample_instance)
                                    all_predictions.append(prediction)
                                final_prediction = np.mean(all_predictions, axis=0)
                            st.write("Prediction for the entered instance:", final_prediction)
                            
                    
                                
                    
                       


            
          


# Sidebar menu
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Data Manipulation", "Characteristics Analysis", "Preprocessing", "Models"])

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
        "Data Summary", "Boxplots and Histograms", 
        "Scatter Plots"
    ])
    if submenu == "Data Summary":
        data_summary_page()
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

elif menu == "Models":
    models_page()
