# app/utils.py

import pandas as pd
import xarray as xr
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def load_netcdf_to_dataframe(file_path: str, lat_fraction=6, lon_fraction=6) -> pd.DataFrame:
    """
    Loads a truncated portion of the NetCDF data into a Pandas DataFrame.
    Truncates both latitude and longitude ranges by the specified fractions.
    """
    try:
        logger.info(f"Opening NetCDF file: {file_path}")
        ds = xr.open_dataset(file_path)
        logger.info("NetCDF file opened successfully.")

        # Get dataset dimensions
        lat_size = ds.dims['lat']
        lon_size = ds.dims['lon']

        # Select a fraction of the data by slicing the latitude and longitude dimensions
        lat_slice = slice(0, lat_size // lat_fraction)
        lon_slice = slice(0, lon_size // lon_fraction)

        logger.info(f"Truncating to 1/{lat_fraction} of latitude and 1/{lon_fraction} of longitude range.")
        logger.info(f"Latitude indices: {lat_slice}, Longitude indices: {lon_slice}")

        # Select the truncated subset of data
        ds_subset = ds.isel(lat=lat_slice, lon=lon_slice)

        # Convert to a DataFrame
        logger.info("Converting the selected subset to a DataFrame...")
        data_df = ds_subset['GWRPM25'].to_dataframe().reset_index()

        # Drop rows with NaN PM2.5 values
        data_df = data_df.dropna(subset=['GWRPM25'])

        # Rename columns for better readability
        data_df = data_df.rename(columns={'lat': 'Latitude', 'lon': 'Longitude', 'GWRPM25': 'PM2.5'})

        # Add an 'id' column
        data_df.reset_index(drop=True, inplace=True)
        data_df['id'] = data_df.index.astype(int)

        # Reorder columns
        data_df = data_df[['id', 'Latitude', 'Longitude', 'PM2.5']]

        # Log the first 5 entries of the DataFrame
        logger.info("First 5 entries of the DataFrame:")
        logger.info(f"\n{data_df.head(5)}")

        logger.info("DataFrame processing completed successfully.")
        return data_df

    except Exception as e:
        logger.exception("An error occurred while loading the NetCDF file.")
        raise e

def get_data_entry_by_id(id: int, data_df: pd.DataFrame) -> Optional[dict]:
    """
    Retrieves a data entry by its ID.
    """
    entry = data_df[data_df['id'] == id]
    if not entry.empty:
        # Convert to native Python types
        result = entry.iloc[0].to_dict()
        result = {k: (v.item() if isinstance(v, (np.generic, np.ndarray)) else v) for k, v in result.items()}
        return result
    else:
        return None

def add_data_entry(new_entry: dict, data_df: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    """
    Adds a new data entry to the DataFrame.
    """
    max_id = data_df['id'].max() if not data_df.empty else -1
    new_id = int(max_id) + 1
    new_entry['id'] = new_id

    # Ensure correct data types
    new_entry['Latitude'] = float(new_entry['Latitude'])
    new_entry['Longitude'] = float(new_entry['Longitude'])
    new_entry['PM2.5'] = float(new_entry['PM2.5'])

    # Create a DataFrame from the new entry
    new_row = pd.DataFrame([new_entry])

    # Concatenate the new row to the existing DataFrame
    updated_data_df = pd.concat([data_df, new_row], ignore_index=True)

    logger.info(f"Added new data entry with ID {new_id}.")
    return new_id, updated_data_df

def update_data_entry(id: int, updated_entry: dict, data_df: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
    """
    Updates an existing data entry.
    """
    if id in data_df['id'].values:
        for key, value in updated_entry.items():
            if key in data_df.columns:
                data_df.loc[data_df['id'] == id, key] = value
        logger.info(f"Updated data entry with ID {id}.")
        return True, data_df
    else:
        return False, data_df

def delete_data_entry(id: int, data_df: pd.DataFrame) -> Tuple[bool, pd.DataFrame]:
    """
    Deletes a data entry from the DataFrame.
    """
    if id in data_df['id'].values:
        data_df = data_df[data_df['id'] != id].reset_index(drop=True)
        logger.info(f"Deleted data entry with ID {id}.")
        return True, data_df
    else:
        return False, data_df

def get_statistics(data_df: pd.DataFrame) -> dict:
    """
    Calculates basic statistics across the dataset.
    """
    stats = {
        "count": int(data_df['PM2.5'].count()),
        "average_pm25": float(data_df['PM2.5'].mean()),
        "min_pm25": float(data_df['PM2.5'].min()),
        "max_pm25": float(data_df['PM2.5'].max()),
    }
    logger.info("Calculated dataset statistics.")
    return stats

def filter_data(data_df: pd.DataFrame, lat: Optional[float], lon: Optional[float]) -> pd.DataFrame:
    """
    Filters the dataset based on latitude and/or longitude.
    """
    filtered_df = data_df
    if lat is not None:
        filtered_df = filtered_df[filtered_df['Latitude'] == lat]
    if lon is not None:
        filtered_df = filtered_df[filtered_df['Longitude'] == lon]
    logger.info("Filtered data based on provided criteria.")
    return filtered_df

def get_data_in_region(data_df: pd.DataFrame, lat_min: float, lat_max: float,
                       lon_min: float, lon_max: float) -> pd.DataFrame:
    """
    Retrieves data within a specified bounding box.
    """
    region_df = data_df[
        (data_df['Latitude'] >= lat_min) & (data_df['Latitude'] <= lat_max) &
        (data_df['Longitude'] >= lon_min) & (data_df['Longitude'] <= lon_max)
    ]
    logger.info("Retrieved data within the specified region.")
    return region_df

def normalize_pm25(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the PM2.5 levels to a scale between 0 and 1.
    """
    pm25_min = data_df['PM2.5'].min()
    pm25_max = data_df['PM2.5'].max()
    if pm25_min == pm25_max:
        raise ValueError("Cannot normalize PM2.5 levels: min and max values are equal")
    data_df_normalized = data_df.copy()
    data_df_normalized['PM2.5_normalized'] = (data_df_normalized['PM2.5'] - pm25_min) / (pm25_max - pm25_min)
    logger.info("Normalized PM2.5 levels to range between 0 and 1.")
    return data_df_normalized[['id', 'Latitude', 'Longitude', 'PM2.5_normalized']]

def get_top10_polluted_locations(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the top 10 most polluted locations in the dataset.
    """
    top10 = data_df.nlargest(10, 'PM2.5')
    logger.info("Retrieved top 10 most polluted locations in the dataset.")
    return top10[['id', 'Latitude', 'Longitude', 'PM2.5']]
