import numpy as np
import pandas as pd

def calculate_ndvi(nir, red):
    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi

def calculate_evi(nir, red, blue):
    # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    evi = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)
    return evi

def extract_phenology_features(time_series):
    # Extract temporal features from vegetation index time series
    features = {
        'max_index': np.max(time_series),
        'min_index': np.min(time_series),
        'mean_index': np.mean(time_series),
        'std_index': np.std(time_series),
        'time_max_index': np.argmax(time_series),
        'time_min_index': np.argmin(time_series),
        'sum_index': np.sum(time_series),
    }
    return features

def normalize_columns(df, columns):
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val + 1e-6)
    return df

def create_features(weather_df, soil_df, satellite_df):
    # Calculate vegetation indices from satellite multispectral data
    satellite_df['NDVI'] = calculate_ndvi(satellite_df['nir'], satellite_df['red'])
    satellite_df['EVI'] = calculate_evi(satellite_df['nir'], satellite_df['red'], satellite_df['blue'])
    
    # Generate phenology features grouped by region over time series (NDVI)
    phenology_df = satellite_df.groupby('region')['NDVI'].apply(
        lambda x: pd.Series(extract_phenology_features(x.values))
    ).reset_index()
    
    # Merge weather, soil, and phenology data
    combined_df = weather_df.merge(soil_df, on='region', how='left')
    combined_df = combined_df.merge(phenology_df, on='region', how='left')
    
    # Select features to normalize
    feature_cols = ['temperature', 'rainfall', 'humidity', 'NDVI', 'EVI', 'organic_matter', 'nitrogen']
    combined_df = normalize_columns(combined_df, feature_cols)
    
    return combined_df

# Example usage:
# weather_df = pd.read_csv('weather_data.csv')
# soil_df = pd.read_csv('soil_data.csv')
# satellite_df = pd.read_csv('satellite_data.csv')  # Should include 'nir', 'red', 'blue' bands and 'region' column
# engineered_df = create_features(weather_df, soil_df, satellite_df)
# engineered_df.to_csv('engineered_features.csv', index=False)
