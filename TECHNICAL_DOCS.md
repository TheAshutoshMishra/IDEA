# AadhaarInsight360 - Technical Documentation

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    AadhaarInsight360                        │
│                  Analytics Platform                          │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼──────┐  ┌────────▼────────┐
│  Data Layer    │  │ Analysis    │  │ Presentation    │
│                │  │ Layer       │  │ Layer           │
│ - Raw Data     │  │ - Patterns  │  │ - Dashboard     │
│ - Processing   │  │ - Anomalies │  │ - Reports       │
│ - Validation   │  │ - Predictions│  │ - APIs          │
└────────────────┘  └─────────────┘  └─────────────────┘
```

## Module Documentation

### 1. Data Processing Module

**File**: `src/data_processing/preprocessor.py`

**Class**: `DataPreprocessor`

#### Methods:

##### `process(data_path: str) -> pd.DataFrame`
Main processing pipeline that orchestrates all data cleaning steps.

**Parameters**:
- `data_path`: Path to raw data file (CSV, Excel, or Parquet)

**Returns**:
- Processed pandas DataFrame ready for analysis

**Process Flow**:
1. Load data from file
2. Remove duplicates
3. Handle missing values
4. Remove outliers
5. Engineer features
6. Save processed data

**Example**:
```python
from src.data_processing import DataPreprocessor

preprocessor = DataPreprocessor(config)
df = preprocessor.process('data/raw/enrollment.csv')
```

##### `_clean_data(df: pd.DataFrame) -> pd.DataFrame`
Cleans raw data by removing duplicates and fixing data types.

##### `_handle_missing_values(df: pd.DataFrame) -> pd.DataFrame`
Handles missing values using median (numeric) and mode (categorical) imputation.

##### `_remove_outliers(df: pd.DataFrame) -> pd.DataFrame`
Removes statistical outliers using IQR or Z-score method.

##### `_engineer_features(df: pd.DataFrame) -> pd.DataFrame`
Creates new features from existing data (time-based, aggregations, etc.)

---

### 2. Pattern Analysis Module

**File**: `src/pattern_analysis/pattern_detector.py`

**Class**: `PatternDetector`

#### Methods:

##### `detect_patterns(df: pd.DataFrame) -> Dict[str, Any]`
Master function that runs all pattern detection analyses.

**Returns**: Dictionary containing:
- `enrollment_patterns`: Enrollment-specific insights
- `update_patterns`: Update behavior analysis
- `geographic_patterns`: Geographic distribution
- `demographic_patterns`: Age, gender demographics
- `temporal_patterns`: Time-based trends
- `clusters`: User segmentation results

**Example**:
```python
from src.pattern_analysis import PatternDetector

detector = PatternDetector(config)
patterns = detector.detect_patterns(df)

print(f"Total enrollments: {patterns['enrollment_patterns']['total_enrollments']}")
print(f"Top state: {list(patterns['geographic_patterns']['top_5_states'].keys())[0]}")
```

##### `analyze_enrollment(df: pd.DataFrame) -> Dict[str, Any]`
Analyzes enrollment-specific patterns including rates, trends, and demographics.

##### `analyze_geographic(df: pd.DataFrame) -> Dict[str, Any]`
Geographic analysis including state/district distribution and urban/rural split.

##### `perform_clustering(df: pd.DataFrame) -> Dict[str, Any]`
K-means clustering to identify user segments and their characteristics.

---

### 3. Anomaly Detection Module

**File**: `src/anomaly_detection/anomaly_detector.py`

**Class**: `AnomalyDetector`

#### Methods:

##### `detect_anomalies(df: pd.DataFrame) -> Dict[str, Any]`
Comprehensive anomaly detection using multiple methods.

**Returns**: Dictionary containing:
- `statistical_outliers`: Z-score and IQR outliers
- `ml_anomalies`: ML-detected anomalies
- `data_quality_issues`: Missing values, duplicates
- `suspicious_patterns`: Behavioral anomalies
- `fraud_indicators`: Fraud risk indicators

**Example**:
```python
from src.anomaly_detection import AnomalyDetector

detector = AnomalyDetector(config)
anomalies = detector.detect_anomalies(df)

print(f"Total anomalies: {anomalies['summary']['total_anomalies_detected']}")
print(f"Anomaly rate: {anomalies['summary']['anomaly_rate']:.2%}")
```

##### `_detect_ml_anomalies(df: pd.DataFrame) -> Dict[str, Any]`
Uses Isolation Forest, LOF, and One-Class SVM for anomaly detection.

**Algorithms**:
1. **Isolation Forest**: Tree-based anomaly detection
2. **Local Outlier Factor (LOF)**: Density-based detection
3. **One-Class SVM**: Support vector machine approach

##### `detect_fraud(df: pd.DataFrame) -> Dict[str, Any]`
Specialized fraud detection including enrollment spikes and inconsistencies.

---

### 4. Predictive Models Module

**File**: `src/predictive_models/forecaster.py`

**Class**: `Forecaster`

#### Methods:

##### `forecast_enrollment(df: pd.DataFrame, periods: int = 12) -> Dict[str, Any]`
Forecasts future enrollment using multiple algorithms.

**Parameters**:
- `df`: Historical enrollment data
- `periods`: Number of time periods to forecast (default: 12)

**Returns**: Dictionary with forecasts from:
- Prophet
- ARIMA
- Moving Average
- Ensemble (average of all methods)

**Example**:
```python
from src.predictive_models import Forecaster

forecaster = Forecaster(config)
predictions = forecaster.forecast_enrollment(df, periods=30)

print(f"30-day forecast: {predictions['ensemble']['predictions'][:5]}")
```

##### `_forecast_with_prophet(ts_data: pd.DataFrame, periods: int) -> Dict[str, Any]`
Facebook Prophet forecasting with seasonality handling.

**Features**:
- Automatic seasonality detection
- Handles holidays and special events
- Provides uncertainty intervals

##### `_forecast_with_arima(ts_data: pd.DataFrame, periods: int) -> Dict[str, Any]`
Statistical ARIMA forecasting.

**Features**:
- Auto-selects best parameters
- Handles trends and seasonality
- Classic time series approach

##### `predict_resource_needs(df: pd.DataFrame) -> Dict[str, Any]`
Predicts resource requirements for enrollment centers.

---

### 5. Visualization Dashboard

**File**: `dashboard/app.py`

**Main Function**: Streamlit-based interactive dashboard

#### Features:

##### Overview Page
- Key metrics (total enrollments, states covered, success rate)
- Enrollment trends
- State-wise distribution
- Age and gender demographics

##### Enrollment Analysis
- Temporal patterns (daily, monthly, yearly)
- Day of week analysis
- Geographic distribution by area type

##### Update Analysis
- Update frequency distribution
- Time to first update
- Update statistics

##### Anomaly Detection
- Age outliers
- Excessive updates
- Biometric failures
- State-wise failure analysis

##### Predictions
- 30-day enrollment forecast
- Resource capacity recommendations
- Trend visualizations

##### Recommendations
- Strategic insights
- Priority-based action items
- Implementation roadmap

**Example Usage**:
```bash
streamlit run dashboard/app.py
```

---

## Configuration

**File**: `config.yaml`

### Key Sections:

#### Data Processing
```yaml
data_processing:
  chunk_size: 50000
  missing_threshold: 0.3
  outlier_method: "IQR"
```

#### Pattern Analysis
```yaml
pattern_analysis:
  clustering:
    method: "kmeans"
    n_clusters: 8
```

#### Anomaly Detection
```yaml
anomaly_detection:
  methods:
    - "isolation_forest"
    - "local_outlier_factor"
  contamination: 0.05
```

#### Predictive Models
```yaml
predictive_models:
  time_series:
    methods:
      - "prophet"
      - "arima"
      - "lstm"
```

---

## Data Schema

### Expected Input Format

#### Enrollment Data
```
enrollment_id, enrollment_date, state, district, age, gender, 
area_type, biometric_status, enrollment_center_id
```

#### Update Data
```
enrollment_id, update_date, update_type, update_count, 
last_update_date
```

### Output Format

#### Analysis Results (JSON)
```json
{
  "metadata": {...},
  "patterns": {...},
  "anomalies": {...},
  "predictions": {...}
}
```

---

## Performance Optimization

### Data Processing
- **Chunk Processing**: Handles large files in chunks
- **Parallel Execution**: Uses multiprocessing where possible
- **Memory Optimization**: Efficient data types

### ML Models
- **Caching**: Results cached for repeated queries
- **Sampling**: Large datasets sampled for expensive algorithms
- **Vectorization**: NumPy operations for speed

---

## Error Handling

### Logging Levels
- **DEBUG**: Detailed information for debugging
- **INFO**: General information messages
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical issues

### Error Recovery
- Fallback algorithms if primary fails
- Graceful degradation
- Detailed error messages in logs

---

## Testing

### Unit Tests
```bash
pytest tests/
```

### Coverage
```bash
pytest --cov=src tests/
```

---

## Deployment

### Local Deployment
```bash
# 1. Setup environment
python setup.py

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run analysis
python src/main_analysis.py

# 4. Launch dashboard
streamlit run dashboard/app.py
```

### Production Considerations
- Use PostgreSQL for large datasets
- Deploy dashboard on cloud (AWS, Azure, GCP)
- Use Redis for caching
- Implement rate limiting
- Add authentication

---

## API Reference (Future Enhancement)

### Proposed REST API Endpoints

```
POST /api/analyze
GET /api/patterns/{analysis_id}
GET /api/anomalies/{analysis_id}
GET /api/predictions/{analysis_id}
GET /api/report/{analysis_id}
```

---

## Troubleshooting

### Common Issues

#### Issue: Module not found
**Solution**: Ensure virtual environment is activated and dependencies installed

#### Issue: Memory error
**Solution**: Reduce chunk_size in config.yaml or process data in smaller batches

#### Issue: Slow performance
**Solution**: Enable multiprocessing in config or reduce data size

---

## Contributing

### Code Style
- Follow PEP 8
- Use type hints
- Document all functions
- Add unit tests

### Pull Request Process
1. Fork repository
2. Create feature branch
3. Add tests
4. Submit PR with description

---

## License

This project is developed for UIDAI Data Hackathon 2026.

---

## Contact & Support

For questions or issues, contact your team lead or refer to project documentation.
