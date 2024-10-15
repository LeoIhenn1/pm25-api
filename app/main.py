from fastapi import FastAPI, HTTPException, Query
from typing import Optional
import logging
import pandas as pd

from app.utils import (
    load_netcdf_to_dataframe,
    get_statistics,
    filter_data,
    get_data_in_region,
    add_data_entry,
    update_data_entry,
    delete_data_entry,
    get_data_entry_by_id,
    normalize_pm25,
    get_top10_polluted_locations,
)
from app.models import DataEntry, DataEntryResponse 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PM2.5 REST API",
    description="API for interacting with PM2.5 data from NetCDF files using Pandas DataFrame.",
    version="1.0.0",
)

# Global DataFrame variable
data_df = pd.DataFrame()

# Load truncated data on startup
@app.on_event("startup")
async def startup_event():
    global data_df
    logger.info("Starting to load NetCDF data.")
    data_df = load_netcdf_to_dataframe("data/global_pm25.nc")
    logger.info("Finished loading NetCDF data.")

# Endpoint to retrieve all data
@app.get("/data", summary="Retrieve all available data")
def get_all_data():
    return data_df.to_dict(orient="records")

# Endpoint to add a new data entry
@app.post("/data", summary="Add a new data entry", response_model=DataEntryResponse)
def add_data(new_entry: DataEntry):
    global data_df
    new_entry_dict = new_entry.dict()
    new_entry_dict['PM2.5'] = new_entry_dict.pop('PM2_5')
    new_id, data_df = add_data_entry(new_entry_dict, data_df)
    return DataEntryResponse(message="Data added successfully", id=new_id)

# Endpoint to provide basic statistics
@app.get("/data/stats", summary="Provide basic statistics across the dataset")
def statistics():
    stats = get_statistics(data_df)
    return stats

# Endpoint to filter data based on latitude and longitude
@app.get("/data/filter", summary="Filter the dataset based on latitude and longitude")
def filter_data_endpoint(
    lat: Optional[float] = Query(None, description="Latitude to filter by"),
    lon: Optional[float] = Query(None, description="Longitude to filter by"),
):
    if lat is None and lon is None:
        raise HTTPException(status_code=400, detail="At least one of 'lat' or 'lon' must be provided")
    filtered = filter_data(data_df, lat, lon)
    if filtered.empty:
        raise HTTPException(status_code=404, detail="No data found for the provided filters")
    return filtered.to_dict(orient="records")

# Endpoint to get data within a bounding box
@app.get("/data/region", summary="Retrieve data within a bounding box")
def data_in_region(
    lat_min: float = Query(..., description="Minimum latitude"),
    lat_max: float = Query(..., description="Maximum latitude"),
    lon_min: float = Query(..., description="Minimum longitude"),
    lon_max: float = Query(..., description="Maximum longitude"),
):
    region_data = get_data_in_region(data_df, lat_min, lat_max, lon_min, lon_max)
    if region_data.empty:
        raise HTTPException(status_code=404, detail="No data found within the specified region")
    return region_data.to_dict(orient="records")


# Endpoint to get data with normalized PM2.5 levels
@app.get("/data/normalized", summary="Get data with normalized PM2.5 levels")
def get_normalized_pm25_endpoint():
    try:
        normalized_df = normalize_pm25(data_df)
        return normalized_df.to_dict(orient='records')
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint to get top 10 polluted locations
@app.get("/data/top10", summary="Get Top 10 most polluted locations in the dataset")
def get_top10_polluted():
    top10 = get_top10_polluted_locations(data_df)
    if top10.empty:
        raise HTTPException(status_code=404, detail="No data available to determine top polluted locations")
    return top10.to_dict(orient='records')

# Endpoint to fetch data by ID
@app.get("/data/{id}", summary="Fetch a specific data entry by ID")
def get_data_by_id_endpoint(id: int):
    data_entry = get_data_entry_by_id(id, data_df)
    if data_entry is not None:
        return data_entry
    else:
        raise HTTPException(status_code=404, detail="Data entry not found")

# Endpoint to delete a data entry
@app.delete("/data/{id}", summary="Delete a data entry")
def delete_data(id: int):
    global data_df
    success, data_df = delete_data_entry(id, data_df)
    if success:
        return {"message": "Data deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Data entry not found")

# Endpoint to update an existing data entry
@app.put("/data/{id}", summary="Update an existing data entry")
def update_data(id: int, updated_entry: DataEntry):
    global data_df
    updated_entry_dict = updated_entry.dict()
    updated_entry_dict['PM2.5'] = updated_entry_dict.pop('PM2_5')
    success, data_df = update_data_entry(id, updated_entry_dict, data_df)
    if success:
        return {"message": "Data updated successfully"}
    else:
        raise HTTPException(status_code=404, detail="Data entry not found")

