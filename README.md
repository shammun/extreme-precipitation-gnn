# Predicting Extreme Precipitation Events with Graph Neural Networks (GNN) and Explainable AI (XAI)

## Overview
This project aims to predict extreme precipitation events using open climate data and advanced machine learning techniques, including Graph Neural Networks (GNNs) and Explainable AI (XAI) methods. Leveraging Google Earth Engine (GEE) for data collection and Google Cloud Storage (GCS) for data handling, I integrated multiple climate datasets to improve the model’s predictive accuracy for extreme rainfall events and explore the drivers behind these phenomena.

---

## Data and Methodology

### 1. Data Collection
We utilize multiple datasets from open sources:

- **Precipitation Data**: GPM IMERG data (NASA)
- **Climate Variables**: ERA5 reanalysis data (temperature, CAPE, surface pressure)
- **Topographical Data**: SRTM Digital Elevation
- **Land Cover**: MODIS land cover types
- **Soil Moisture**: SMAP surface soil moisture

Data is collected using GEE, filtered for our region of interest, and preprocessed to a structured format in Google Cloud Storage. We define extreme precipitation events based on the 99th percentile of the historical precipitation data. The data was taken from the GEE dataset and preprocessed to a structured format in Google Cloud Storage. I considered the monsoon season as the period from June to September.

```python
# Define Area of Interest (AOI) and date range
aoi = ee.Geometry.Rectangle([longitude_min, latitude_min, longitude_max, latitude_max])
# Define the time frame
start_year = 2015
end_year = 2019
# Define monsoon months
monsoon_months = [6, 7, 8, 9]

# Get the Bangladesh boundary
# Import the country boundaries dataset
countries = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017')

# Filter for Bangladesh
bangladesh = countries.filter(ee.Filter.eq('country_na', 'Bangladesh')).geometry()

# Access GPM IMERG precipitation data
gpm_dataset = ee.ImageCollection('NASA/GPM_L3/IMERG_V06').filterDate(start_date, end_date).filterBounds(aoi)
precipitation = gpm_dataset.select('precipitationCal')

# Access ERA5 reanalysis data
era5_dataset = ee.ImageCollection('ECMWF/ERA5/DAILY').select([
    'mean_2m_air_temperature',
    'minimum_2m_air_temperature',
    'maximum_2m_air_temperature',
    'dewpoint_2m_temperature',
    'total_precipitation',
    'surface_pressure',
    'mean_sea_level_pressure',
    'u_component_of_wind_10m',
    'v_component_of_wind_10m'
])

# Load SRTM Digital Elevation Data
srtm_dataset = ee.Image('USGS/SRTMGL1_003').filterBounds(aoi)

# Load MODIS Land Cover for each year
modis_landcover = {}
for year in range(start_year, end_year + 1):
    lc_image = ee.ImageCollection('MODIS/006/MCD12Q1') \
                .filterDate(f'{year}-01-01', f'{year}-12-31') \
                .first() \
                .select('LC_Type1') \
                .clip(bangladesh)
    modis_landcover[str(year)] = lc_image

# Load SMAP Surface Soil Moisture
smap_dataset = ee.ImageCollection('NASA_USDA/HSL/SMAP10KM_soil_moisture') \
                  .filterBounds(bangladesh) \
                  .select('ssm')

# Load SMAP Surface Soil Moisture
smap_dataset = ee.ImageCollection('NASA_USDA/HSL/SMAP10KM_soil_moisture') \
                  .filterBounds(bangladesh) \
                  .select('ssm')
```

### 2. Data Enginnering

After data export from GEE to Google Cloud Storage, data cleaning and feature engineering are performed:

The following steps were taken to prepare the data for modeling:

- **Data Cleaning**: Interpolated missing values in the time series and used imputation for spatial gaps.
  - Interpolate missing 'ssm' values per location over time.
    To handle missing values in ssm, the data was first sorted by location (longitude and latitude) and date. Linear interpolation was applied within each location’s time series to estimate missing values based on nearby points. Any remaining gaps were then filled using forward-fill and backward-fill, ensuring continuity across the dataset without any missing values in ssm.

    ```python
    # Ensure 'date' is in datetime format
    data['date'] = pd.to_datetime(data['date'])

    # Sort data by location and date
    data.sort_values(['longitude', 'latitude', 'date'], inplace=True)

    # Interpolate 'ssm' per location using groupby().transform()
    data['ssm_interpolated'] = data.groupby(['longitude', 'latitude'])['ssm'].transform(
        lambda group: group.interpolate(method='linear')
    )

    # Forward-fill and backward-fill remaining NaN values per location
    data['ssm_interpolated'] = data.groupby(['longitude', 'latitude'])['ssm_interpolated'].transform(
        lambda group: group.fillna(method='ffill').fillna(method='bfill')
    )
    ```

  - Dealing with issing values in land cover data:
    For each location (grouped by longitude and latitude), missing values were filled using the mode (most frequent value) within each group. Any remaining gaps were filled with forward-fill and backward-fill.
      
    ```python
    # Convert 'LC_Type1' to integer (handling missing values)
    data['LC_Type1'] = data['LC_Type1'].astype('Int64')
    # Define a function to fill missing 'LC_Type1' values per group
    def fill_lc_type(group):
        # Fill with mode
        if not group.mode().empty:
            group = group.fillna(group.mode().iloc[0])
        else:
            group = group
        # Then forward-fill and backward-fill
        group = group.ffill().bfill()
        return group

    # Apply the function using groupby().transform()
    data['LC_Type1_filled'] = data.groupby(['longitude', 'latitude'])['LC_Type1'].transform(fill_lc_type)
    ```
  - Missing vakues in elevation data:
    Each location (grouped by longitude and latitude) was processed individually. Missing elevation values were filled using forward-fill and backward-fill within each group.

    ```python
    data['elevation_filled'] = data.groupby(['longitude', 'latitude'])['elevation'].transform(
    lambda group: group.fillna(method='ffill').fillna(method='bfill'))
    ```
    
  - Missing values in other numerical variables:
    K nearest neighbors (KNN) imputation was used to fill missing values in other numerical variables.

    ```python
    # List of variables to impute
    knn_impute_vars = [
        'dewpoint_2m_temperature', 'maximum_2m_air_temperature', 'mean_2m_air_temperature',
        'mean_sea_level_pressure', 'minimum_2m_air_temperature', 'surface_pressure',
        'total_precipitation', 'u_component_of_wind_10m', 'v_component_of_wind_10m'
    ]

    # Initialize KNN Imputer
    knn_imputer = KNNImputer(n_neighbors=5)

    # Impute missing values
    data[knn_impute_vars] = knn_imputer.fit_transform(data[knn_impute_vars])
    ```

- **Data Merging**: Merged multiple data sources (precipitation, climate, topography, land cover, soil moisture) and aligned them temporally and spatially.

- **Feature Engineering:**
  - **Temporal Features**: Created features such as day of the year, month, and season.
  - **Spatial Features**: Added latitude, longitude, and elevation.
  - **Derived Variables**: Calculated anomalies, such as CAPE (Convective Available Potential Energy), for capturing atmospheric conditions.


Data is split into training and validation sets, with features scaled for optimal model performance.

```python
# Define extreme precipitation threshold
threshold = data_df['precipitation'].quantile(0.99)
data_df['extreme_event'] = (data_df['precipitation'] > threshold).astype(int)
```

- **Label Creation**: Defined extreme precipitation events as instances where precipitation exceeded the 99th percentile. Created a binary label for model training.

- **Scaling**: Applied Min-Max scaling and standardization for consistent feature ranges.

  ```python
  from sklearn.preprocessing import StandardScaler
  # Initialize StandardScaler
  scaler = StandardScaler()

  # Apply StandardScaler to numerical columns
  data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
  ```

### 3. Model Development
Using PyTorch and PyTorch Lightning, I trained several models, including Temporal Fusion Transformer (TFT) and Graph Neural Networks (GNN), for capturing spatial and temporal patterns in the data. The GNN outperformed other models in accurately identifying extreme precipitation events.

#### Temporal Fusion Transformer (TFT)
The Temporal Fusion Transformer (TFT) was selected due to its ability to handle time series data and extract temporal patterns. The TFT uses attention mechanisms to quantify feature importance and capture dependencies across time steps and features.

```python
import pytorch_lightning as pl

class TemporalFusionTransformerLightningModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = self.model.hparams.loss  # Use the updated loss function
        self.save_hyperparameters(ignore=['model', 'loss'])

    def forward(self, x):
        output = self.model(x)
        return output['prediction']  # Return only the prediction tensor

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = recursive_to_device(x, self.device)
        y = recursive_to_device(y[0], self.device).long()
        y_pred = self(x)
        loss = self.loss(y_pred.view(-1, y_pred.size(-1)), y.view(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        print(f"Epoch {self.current_epoch}, Training Loss: {loss.item()}")
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = recursive_to_device(x, self.device)
        y = recursive_to_device(y[0], self.device).long()
        y_pred = self(x)
        loss = self.loss(y_pred.view(-1, y_pred.size(-1)), y.view(-1))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        print(f"Epoch {self.current_epoch}, Validation Loss: {loss.item()}")
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if isinstance(batch, tuple) or isinstance(batch, list):
            x = batch[0]
        else:
            x = batch
        x = recursive_to_device(x, self.device)
        y_pred = self(x)
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.model.hparams.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.model.hparams.reduce_on_plateau_patience,
            factor=0.5,
            verbose=True,  # Set verbose to True
        )
        scheduler_config = {
            'scheduler': scheduler,
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1,
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}
```
After training the model, I found that it was not detecting any extreme precipitation events. This was likely due to the model's inability to capture the complex spatial and temporal patterns in the data.

I used focal loss to address this issue. Focal loss is a variant of the cross-entropy loss that down-weights the loss assigned to well-classified examples, thereby focusing the model's attention on the misclassified examples. But that didn't work either. Thus, I decided to use a Spatio-Temporal Graph Neural Network (GNN) to capture the spatial and temporal dependencies in the data.

#### Spatio-Temporal Graph Neural Network (GNN)
The Spatio-Temporal GNN model was implemented to capture both spatial and temporal dependencies by structuring the data as a graph, where each node represents a geographical region with connections to neighboring nodes. This structure allows the model to consider the influence of neighboring regions on extreme precipitation events.

I constructed a spatial graph where each node represents a location, and edges are defined based on geographical proximity. 

```python
class SpatioTemporalGNN(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(SpatioTemporalGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Node-level predictions (raw logits)
        x = self.fc(x)

        return x  # No activation; handled by loss function

# Initialize the model
model = SpatioTemporalGNN(num_node_features=num_node_features, num_classes=num_classes).to(device)
```

After training the model, now this  model was able to detect extreme precipitation events. Please, have a look at the confusion matrix below.

![Confusion matrix](./1.png)

### Explainable AI with SHAP
Using SHAP (GraphLIME) for interpretability, I examined the feature importance for the Spatio-Temporal GNN models. SHAP helped identify key variables influencing extreme precipitation predictions, such as temperature at 2m above the surface, total precipitation and day of the month, offering insights into the atmospheric drivers behind heavy rainfall events.










