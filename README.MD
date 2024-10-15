# PM2.5 REST API
---

## Overview

The **PM2.5 REST API** is a robust web service built using FastAPI that allows users to interact with PM2.5 pollution data sourced from NetCDF files. The API leverages Pandas for data manipulation and provides comprehensive endpoints for data retrieval, addition, update, deletion, filtering, normalization, and statistical analysis.

## Features

- **Data Retrieval:** Fetch all PM2.5 data entries or specific entries by ID.
- **Data Manipulation:** Add, update, or delete PM2.5 data entries.
- **Statistical Analysis:** Obtain basic statistics like count, average, minimum, and maximum PM2.5 levels.
- **Filtering:** Filter data based on latitude and/or longitude.
- **Region-Based Queries:** Retrieve data within specified geographic bounding boxes.
- **Data Normalization:** Normalize PM2.5 levels to a 0-1 scale.
- **Top Polluted Locations:** Identify the top 10 most polluted locations in the dataset.
- **Comprehensive Testing:** Ensures reliability and correctness of all functionalities.

## Technology Stack

- **Backend Framework:** [FastAPI](https://fastapi.tiangolo.com/)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/)
- **NetCDF Handling:** [Xarray](http://xarray.pydata.org/en/stable/)
- **Testing Framework:** [Pytest](https://pytest.org/)
- **Type Hinting and Validation:** [Pydantic](https://pydantic-docs.helpmanual.io/)
- **JSON Serialization:** Python's built-in `json` module

## Installation

### Prerequisites

- **Python 3.10 or higher:** Ensure Python is installed on your system. You can download it from [here](https://www.python.org/downloads/).

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/pm25-rest-api.git
   cd pm25-rest-api

2. **Use ./run build command to build your docker image:**
   
   Ensure this is run while in the root directory

    ```bash
    ./run build

3. **Use ./run up command to run the app inside your docker image:**

    Ensure this is run while in the root directory

    ```bash
    ./run up

4. **Use ./run restart command to restart the app inside your docker image:**

    Ensure this is run while in the root directory

    ```bash
    ./run restart

    use this after rebuilding the app for any reason

5. **Use ./run test command to run the test suite inside your docker image:**
   
    Ensure this is run while in the root directory

    ```bash
    ./run test

## API Documentation 
With the container running, you can access the API endpoints via your web browser or tools like curl or Postman.

Base URL: http://localhost:8000

Swagger UI Documentation: http://localhost:8000/docs

ReDoc Documentation: http://localhost:8000/redoc

Please refer to these documents for a full preview of the API.
