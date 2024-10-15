import pytest
from fastapi.testclient import TestClient
from app.main import app
from app import utils
import pandas as pd

# Create a TestClient using the FastAPI app
client = TestClient(app)

# Fixtures for test data
@pytest.fixture(scope="module")
def test_client():
    # Setup before tests
    with TestClient(app) as c:
        yield c  # Testing happens here

# Helper function to get the current data size
def get_data_size():
    response = client.get("/data")
    return len(response.json())

# -------------------------------
# Testing Utility Functions
# -------------------------------

def test_load_netcdf_to_dataframe():
    # Assuming there is a test NetCDF file available at 'data/test_pm25.nc'
    try:
        data_df = utils.load_netcdf_to_dataframe("data/global_pm25.nc", lat_fraction=6, lon_fraction=6)
        assert isinstance(data_df, pd.DataFrame)
        assert not data_df.empty
        assert set(['id', 'Latitude', 'Longitude', 'PM2.5']).issubset(data_df.columns)
    except FileNotFoundError:
        pytest.skip("NetCDF test file not found")

def test_get_statistics():
    # Create a sample DataFrame
    data = {'PM2.5': [10, 20, 30, 40, 50]}
    data_df = pd.DataFrame(data)
    stats = utils.get_statistics(data_df)
    assert stats['count'] == 5
    assert stats['average_pm25'] == 30.0
    assert stats['min_pm25'] == 10.0
    assert stats['max_pm25'] == 50.0

def test_filter_data():
    # Create a sample DataFrame
    data = {
        'id': [0, 1, 2],
        'Latitude': [10.0, 20.0, 10.0],
        'Longitude': [30.0, 40.0, 30.0],
        'PM2.5': [15.0, 25.0, 35.0]
    }
    data_df = pd.DataFrame(data)
    filtered_df = utils.filter_data(data_df, lat=10.0, lon=None)
    assert len(filtered_df) == 2
    for index, row in filtered_df.iterrows():
        assert row['Latitude'] == 10.0

def test_get_data_in_region():
    # Create a sample DataFrame
    data = {
        'id': [0, 1, 2],
        'Latitude': [10.0, 20.0, 15.0],
        'Longitude': [30.0, 40.0, 35.0],
        'PM2.5': [15.0, 25.0, 35.0]
    }
    data_df = pd.DataFrame(data)
    region_df = utils.get_data_in_region(data_df, lat_min=10.0, lat_max=20.0, lon_min=30.0, lon_max=40.0)
    assert len(region_df) == 3

def test_add_data_entry():
    # Create a sample DataFrame
    data_df = pd.DataFrame(columns=['id', 'Latitude', 'Longitude', 'PM2.5'])
    new_entry = {'Latitude': 10.0, 'Longitude': 20.0, 'PM2.5': 15.0}
    new_id, updated_df = utils.add_data_entry(new_entry, data_df)
    assert new_id == 0
    assert len(updated_df) == 1
    assert updated_df.iloc[0]['PM2.5'] == 15.0

def test_update_data_entry():
    # Create a sample DataFrame
    data = {'id': [0], 'Latitude': [10.0], 'Longitude': [20.0], 'PM2.5': [15.0]}
    data_df = pd.DataFrame(data)
    updated_entry = {'Latitude': 12.0, 'Longitude': 22.0, 'PM2.5': 18.0}
    success, updated_df = utils.update_data_entry(0, updated_entry, data_df)
    assert success
    assert updated_df.loc[updated_df['id'] == 0, 'PM2.5'].values[0] == 18.0

def test_delete_data_entry():
    # Create a sample DataFrame
    data = {'id': [0], 'Latitude': [10.0], 'Longitude': [20.0], 'PM2.5': [15.0]}
    data_df = pd.DataFrame(data)
    success, updated_df = utils.delete_data_entry(0, data_df)
    assert success
    assert len(updated_df) == 0

def test_get_data_entry_by_id():
    # Create a sample DataFrame
    data = {'id': [0], 'Latitude': [10.0], 'Longitude': [20.0], 'PM2.5': [15.0]}
    data_df = pd.DataFrame(data)
    entry = utils.get_data_entry_by_id(0, data_df)
    assert entry is not None
    assert entry['PM2.5'] == 15.0

def test_normalize_pm25():
    # Create a sample DataFrame
    data = {'id': [0, 1, 2], 'Latitude': [10.0, 20.0, 30.0], 'Longitude': [40.0, 50.0, 60.0], 'PM2.5': [10.0, 20.0, 30.0]}
    data_df = pd.DataFrame(data)
    normalized_df = utils.normalize_pm25(data_df)
    assert 'PM2.5_normalized' in normalized_df.columns
    assert normalized_df['PM2.5_normalized'].iloc[0] == 0.0
    assert normalized_df['PM2.5_normalized'].iloc[2] == 1.0

def test_get_top10_polluted_locations():
    # Create a sample DataFrame
    data = {
        'id': list(range(15)),
        'Latitude': [float(i) for i in range(15)],
        'Longitude': [float(i) for i in range(15)],
        'PM2.5': [float(i) for i in range(15)]
    }
    data_df = pd.DataFrame(data)
    top10_df = utils.get_top10_polluted_locations(data_df)
    assert len(top10_df) == 10
    assert top10_df['PM2.5'].tolist() == list(range(14, 4, -1))

# -------------------------------
# Testing API Endpoints
# -------------------------------

def test_get_all_data(test_client):
    response = test_client.get("/data")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0  # Should return data entries
    required_keys = {'id', 'Latitude', 'Longitude', 'PM2.5'}
    assert required_keys.issubset(data[0].keys())

def test_get_data_by_id(test_client):
    # Get the first data entry
    response = test_client.get("/data")
    data = response.json()
    first_id = data[0]['id']

    # Fetch data by ID
    response = test_client.get(f"/data/{first_id}")
    assert response.status_code == 200
    data_entry = response.json()
    assert data_entry['id'] == first_id

def test_get_data_by_invalid_id(test_client):
    invalid_id = -1
    response = test_client.get(f"/data/{invalid_id}")
    assert response.status_code == 404
    assert response.json()['detail'] == "Data entry not found"

def test_add_new_data_entry(test_client):
    new_entry = {
        "Latitude": 10.0,
        "Longitude": 20.0,
        "PM2_5": 15.5
    }
    response = test_client.post("/data", json=new_entry)
    assert response.status_code == 200
    result = response.json()
    assert "message" in result and result["message"] == "Data added successfully"
    assert "id" in result

    # Verify the new entry is in the dataset
    new_id = result["id"]
    response = test_client.get(f"/data/{new_id}")
    assert response.status_code == 200
    data_entry = response.json()
    assert data_entry["Latitude"] == new_entry["Latitude"]
    assert data_entry["Longitude"] == new_entry["Longitude"]
    assert data_entry["PM2.5"] == new_entry["PM2_5"]

def test_add_data_missing_fields(test_client):
    incomplete_entry = {
        "Latitude": 10.0,
        "PM2_5": 15.5
    }
    response = test_client.post("/data", json=incomplete_entry)
    assert response.status_code == 422
    # The exact error message might vary; we can check for 'field required' in any of the errors
    errors = response.json()['detail']
    assert any(error['msg'] == 'field required' for error in errors)

def test_update_data_entry(test_client):
    # Add a new entry to update
    new_entry = {
        "Latitude": 30.0,
        "Longitude": 40.0,
        "PM2_5": 25.5
    }
    response = test_client.post("/data", json=new_entry)
    new_id = response.json()["id"]

    # Update the entry
    updated_entry = {
        "Latitude": 30.0,
        "Longitude": 40.0,
        "PM2_5": 35.0
    }
    response = test_client.put(f"/data/{new_id}", json=updated_entry)
    assert response.status_code == 200
    assert response.json()["message"] == "Data updated successfully"

    # Verify the update
    response = test_client.get(f"/data/{new_id}")
    data_entry = response.json()
    assert data_entry["PM2.5"] == updated_entry["PM2_5"]

def test_update_invalid_data_entry(test_client):
    invalid_id = -1
    updated_entry = {
        "Latitude": 50.0,
        "Longitude": 60.0,
        "PM2_5": 35.0
    }
    response = test_client.put(f"/data/{invalid_id}", json=updated_entry)
    assert response.status_code == 404
    assert response.json()["detail"] == "Data entry not found"

def test_delete_data_entry(test_client):
    # Add a new entry to delete
    new_entry = {
        "Latitude": 50.0,
        "Longitude": 60.0,
        "PM2_5": 45.5
    }
    response = test_client.post("/data", json=new_entry)
    new_id = response.json()["id"]

    # Delete the entry
    response = test_client.delete(f"/data/{new_id}")
    assert response.status_code == 200
    assert response.json()["message"] == "Data deleted successfully"

    # Verify deletion
    response = test_client.get(f"/data/{new_id}")
    assert response.status_code == 404

def test_delete_invalid_data_entry(test_client):
    invalid_id = -1
    response = test_client.delete(f"/data/{invalid_id}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Data entry not found"

def test_get_statistics(test_client):
    response = test_client.get("/data/stats")
    assert response.status_code == 200
    stats = response.json()
    required_keys = {"count", "average_pm25", "min_pm25", "max_pm25"}
    assert required_keys.issubset(stats.keys())

def test_filter_data_endpoint(test_client):
    # Use known Latitude and Longitude from existing data
    response = test_client.get("/data")
    data = response.json()
    if len(data) == 0:
        pytest.skip("No data available to test filtering.")
    test_lat = data[0]["Latitude"]
    test_lon = data[0]["Longitude"]

    response = test_client.get(f"/data/filter?lat={test_lat}&lon={test_lon}")
    assert response.status_code == 200
    filtered_data = response.json()
    assert len(filtered_data) > 0
    for entry in filtered_data:
        assert entry["Latitude"] == test_lat
        assert entry["Longitude"] == test_lon

def test_get_data_in_region_endpoint(test_client):
    # Define a region that includes some data points
    response = test_client.get("/data")
    data = response.json()
    if len(data) == 0:
        pytest.skip("No data available to test data in region.")
    latitudes = [entry['Latitude'] for entry in data]
    longitudes = [entry['Longitude'] for entry in data]
    lat_min = min(latitudes)
    lat_max = max(latitudes)
    lon_min = min(longitudes)
    lon_max = max(longitudes)

    response = test_client.get(f"/data/region?lat_min={lat_min}&lat_max={lat_max}&lon_min={lon_min}&lon_max={lon_max}")
    assert response.status_code == 200
    region_data = response.json()
    assert len(region_data) > 0
    for entry in region_data:
        assert lat_min <= entry["Latitude"] <= lat_max
        assert lon_min <= entry["Longitude"] <= lon_max

def test_get_data_in_empty_region(test_client):
    # Define a region with no data points
    lat_min = 1000
    lat_max = 1001
    lon_min = 1000
    lon_max = 1001

    response = test_client.get(f"/data/region?lat_min={lat_min}&lat_max={lat_max}&lon_min={lon_min}&lon_max={lon_max}")
    assert response.status_code == 404
    assert response.json()['detail'] == "No data found within the specified region"

def test_get_normalized_pm25_endpoint(test_client):
    response = test_client.get("/data/normalized")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    for entry in data:
        assert 'PM2.5_normalized' in entry
        assert 0.0 <= entry['PM2.5_normalized'] <= 1.0

def test_get_top10_polluted(test_client):
    response = test_client.get("/data/top10")
    assert response.status_code == 200
    data = response.json()
    assert len(data) <= 10
    pm25_values = [entry['PM2.5'] for entry in data]
    assert pm25_values == sorted(pm25_values, reverse=True)


