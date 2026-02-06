# CarbonCheck Field - Technical Overview

## Executive Summary

CarbonCheck Field is an end-to-end ML-powered mobile application that analyzes agricultural fields using satellite imagery to classify crop types and estimate carbon credit income. The system demonstrates production-grade MLOps practices including automated training pipelines, multi-model serving, label-free drift detection, and comprehensive observability—all built on Google Cloud Platform.

**Key Technical Highlights:**
- Multi-model inference (Random Forest + TensorFlow DNN) with automatic best-model selection
- Label-free data drift detection using statistical methods (KS, PSI, Jensen-Shannon)
- Automated retraining pipeline with champion/challenger evaluation
- Native Vertex AI TensorBoard integration for experiment tracking
- Cross-platform Flutter app (iOS, Android, Web) with real-time satellite imagery

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                          │
│  │  iOS App    │  │ Android App │  │   Web App   │  Flutter + Google Maps   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                          │
└─────────┼────────────────┼────────────────┼─────────────────────────────────┘
          │                │                │
          └────────────────┼────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API LAYER (Cloud Run)                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  FastAPI Backend                                                     │    │
│  │  • Firebase Auth verification                                        │    │
│  │  • Multi-model inference (RF + DNN)                                  │    │
│  │  • Grid-based field analysis for multi-crop detection                │    │
│  │  • Carbon credit calculation                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┬────────────────┐
          ▼                ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Earth Engine │  │  Vertex AI   │  │   BigQuery   │  │ Cloud Storage│
│              │  │              │  │              │  │              │
│ • NDVI data  │  │ • Model      │  │ • Training   │  │ • Model      │
│ • Sentinel-2 │  │   serving    │  │   features   │  │   artifacts  │
│ • CDL labels │  │ • TensorBoard│  │ • Metrics    │  │ • Configs    │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

---

## GCP Services Utilized

| Service | Purpose | Implementation Details |
|---------|---------|----------------------|
| **Vertex AI** | Model training & serving | Custom training containers, TensorBoard integration, experiment tracking |
| **Cloud Run** | API hosting | Serverless FastAPI backend, auto-scaling, 300s timeout for large fields |
| **BigQuery** | Data warehouse | Training data storage, metrics tracking, deployment history |
| **Earth Engine** | Satellite data | NDVI feature extraction from Sentinel-2, CDL crop verification |
| **Cloud Storage** | Artifact storage | Model checkpoints, configuration, TensorBoard logs |
| **Cloud Build** | CI/CD | Automated Docker builds, multi-stage deployments |
| **Cloud Scheduler** | Job orchestration | Monthly drift detection triggers |
| **Cloud Functions** | Event triggers | Lightweight job triggers for skew detection |
| **Artifact Registry** | Container registry | Training and inference container images |

---

## MLOps Pipeline

### Training Pipeline Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Collect   │    │   Train     │    │  Evaluate   │    │   Deploy    │
│    Data     │───▶│   Models    │───▶│  (Quality   │───▶│  (Champion/ │
│             │    │  (RF + DNN) │    │   Gates)    │    │  Challenger)│
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                  │                  │                  │
      ▼                  ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Earth Engine│    │  Vertex AI  │    │ TensorBoard │    │ Cloud Run   │
│ → BigQuery  │    │Custom Job   │    │ Metrics     │    │ Hot Reload  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Key MLOps Features

1. **Dynamic Code Mounting**: Training code is uploaded to GCS at runtime, allowing code changes without Docker rebuilds
2. **Champion/Challenger Evaluation**: New models must exceed quality gates before replacing production models
3. **Quality Gates**: Configurable thresholds for accuracy, per-class F1, and improvement margins
4. **Experiment Tracking**: All training runs logged to Vertex AI TensorBoard with metrics, confusion matrices, and feature importance visualizations

### Quality Gate Configuration
```yaml
quality_gates:
  absolute_min_accuracy: 0.85
  min_per_crop_f1: 0.70
  improvement_margin: 0.02  # 2% better than champion
  min_training_samples: 100
  min_test_samples: 50
```

---

## Multi-Model Architecture

The system implements a pluggable model registry pattern supporting multiple model types:

### Model Registry Design Pattern

```python
class BaseModel(ABC):
    """Abstract base class for all models."""
    
    @abstractmethod
    def train(self, X, y) -> Dict[str, Any]: ...
    
    @abstractmethod
    def predict(self, X) -> np.ndarray: ...
    
    @abstractmethod
    def predict_proba(self, X) -> np.ndarray: ...
    
    @abstractmethod
    def save(self, path: str) -> None: ...
    
    @abstractmethod
    def load(self, path: str) -> None: ...
```

### Implemented Models

| Model | Framework | Hyperparameter Tuning | Use Case |
|-------|-----------|----------------------|----------|
| **Random Forest** | scikit-learn | Grid search | Interpretable baseline, feature importance |
| **Deep Neural Network** | TensorFlow/Keras | Bayesian (Keras Tuner) | Higher capacity, complex patterns |

### Inference Strategy
- Both models run inference in parallel
- Response includes predictions from each model
- Best model selected based on confidence score
- Enables A/B comparison and ensemble potential

---

## Data Pipeline

### Feature Engineering

Features are extracted from **Sentinel-2 satellite imagery** via Google Earth Engine:

| Feature | Description | Temporal Aspect |
|---------|-------------|-----------------|
| `ndvi_mean` | Average NDVI across growing season | Apr 15 - Sep 1 |
| `ndvi_std` | NDVI variability | Full season |
| `ndvi_min/max` | NDVI range | Full season |
| `ndvi_p25/p50/p75` | Quartile distribution | Full season |
| `ndvi_early` | Early season greenup | Apr 15 - Jun 1 |
| `ndvi_late` | Late season maturity | Jul 1 - Sep 1 |

### Data Collection Strategy
- **300 fields per crop type** sampled across multiple counties
- **3 samples per field** for spatial stability
- **CDL verification** ensures label accuracy
- **"Other" category** includes non-crop areas (buildings, roads, water) for robust classification

### Geographic Coverage
- **Corn/Soybeans**: Illinois, Indiana, Iowa, Minnesota
- **Alfalfa**: California, Washington, Idaho, Colorado
- **Winter Wheat**: Kansas, Oklahoma, Texas, Nebraska

---

## Drift Detection System

### Label-Free Monitoring

The drift detection system monitors for distribution shifts **without requiring new labeled data**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DRIFT DETECTION PIPELINE                      │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Sample Fresh │    │  Compare     │    │  Compute     │       │
│  │ Data from    │───▶│  Against     │───▶│  Drift       │       │
│  │ Earth Engine │    │  Training    │    │  Score       │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                             │                    │               │
│                             ▼                    ▼               │
│                      ┌─────────────────────────────────┐        │
│                      │    Statistical Methods           │        │
│                      │  • KS Test (any shift)           │        │
│                      │  • PSI (population stability)    │        │
│                      │  • JS Divergence (symmetric)     │        │
│                      │  • Skewness/Kurtosis drift       │        │
│                      │  • Tail distribution changes     │        │
│                      └─────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### Drift Score Composition

| Component | Weight | What It Detects |
|-----------|--------|-----------------|
| Skewness & Kurtosis | 40% | Distribution shape changes |
| KS/PSI/JS | 30% | Overall distribution shift |
| Tail Drift | 20% | Extreme value changes |
| Prediction Entropy | 10% | Model uncertainty increase |

### Automated Response
- **Score < 30**: No action needed
- **Score 30-50**: Monitor closely
- **Score 50-70**: Consider retraining
- **Score > 70**: Retraining recommended

---

## Deployment Architecture

### Container Strategy

| Container | Registry | Purpose |
|-----------|----------|---------|
| `carboncheck-field-api` | GCR | Production API (Cloud Run) |
| `ml-pipeline` | GCR | Pipeline orchestrator (Cloud Run) |
| `crop-trainer` | Artifact Registry | Vertex AI training |
| `skew-audit` | Artifact Registry | Drift detection jobs |

### Infrastructure as Code

Configuration is managed via YAML and stored in Cloud Storage:

```yaml
# config.yaml - Single source of truth
model:
  rf:
    n_estimators: 100
    max_depth: 15
  dnn:
    hidden_layers: [64, 32, 16]
    dropout_rate: 0.3
    use_tuner: true
    tuner_trials: 15
```

**Hot reload**: Configuration changes take effect without redeployment—just upload new config to GCS.

---

## Observability

### TensorBoard Integration

All training and drift detection runs log to **Vertex AI TensorBoard**:

- **Training Metrics**: Accuracy, F1, precision, recall per class
- **Visualizations**: Confusion matrices, feature importance, learning curves
- **Drift Monitoring**: Distribution comparisons, Q-Q plots, skewness charts
- **Model Comparison**: Side-by-side RF vs DNN performance

### Logging Architecture

```
Training Job
    │
    ├── Scalars: accuracy, loss, f1_score
    ├── Images: confusion_matrix, feature_importance
    └── Histograms: prediction_distribution

Drift Detection
    │
    ├── Scalars: drift_score, ks_statistic, psi
    ├── Images: distribution_comparison, qq_plots
    └── Text: retraining_recommendation
```

---

## Technology Stack

### Languages & Frameworks
| Layer | Technology |
|-------|------------|
| Mobile/Web | Flutter (Dart) |
| Backend API | Python, FastAPI |
| ML Training | Python, TensorFlow, scikit-learn |
| Infrastructure | Docker, Cloud Build |

### Key Libraries
- **ML**: TensorFlow 2.x, Keras Tuner, scikit-learn, NumPy, Pandas
- **GCP**: google-cloud-aiplatform, google-cloud-bigquery, earthengine-api
- **API**: FastAPI, uvicorn, gunicorn
- **Visualization**: matplotlib, TensorBoard

### DevOps Tools
- **CI/CD**: Cloud Build, GitHub Actions (Vercel)
- **Containers**: Docker, Artifact Registry
- **Scheduling**: Cloud Scheduler, Cloud Functions

---

## Key Engineering Decisions

| Decision | Rationale |
|----------|-----------|
| **Multi-model serving** | Enables model comparison, graceful fallback, future ensemble |
| **Label-free drift detection** | Production data rarely has labels; statistical methods scale |
| **Dynamic code mounting** | Rapid iteration without Docker rebuilds |
| **Grid-based field analysis** | Large fields may contain multiple crops |
| **Earth Engine for features** | Consistent, globally available satellite data |
| **FastAPI over Flask** | Async support, automatic OpenAPI docs, type hints |
| **TensorBoard over custom dashboards** | Native Vertex AI integration, industry standard |

---

## Metrics & Scale

| Metric | Value |
|--------|-------|
| Training data | ~5,000 samples across 5 crop types |
| Feature dimensions | 9 NDVI-derived features |
| Inference latency | <2s for single prediction |
| Grid analysis | Up to 100+ cells for large fields |
| Drift detection frequency | Monthly automated runs |

---

## Future Enhancements

- **Ensemble methods**: Combine RF and DNN predictions
- **Active learning**: Prioritize uncertain samples for labeling
- **Continuous training**: Trigger retraining on drift detection
- **A/B testing framework**: Route traffic between model versions
- **Edge deployment**: TFLite models for offline mobile inference

---

## Repository Structure

```
carbon_check_field/
├── lib/                    # Flutter app (Dart)
├── backend/                # FastAPI inference API
├── ml_pipeline/
│   ├── orchestrator/       # Pipeline coordination
│   ├── trainer/            # Vertex AI training code
│   │   └── models/         # Pluggable model registry
│   ├── skew_job/           # Drift detection
│   ├── shared/             # Common utilities
│   └── tools/              # Development utilities
├── docs/                   # Documentation
└── scripts/                # Build & deployment scripts
```

---

## Contact

This project demonstrates end-to-end ML system design, from data collection through production monitoring. For questions about architecture decisions or implementation details, the codebase is fully documented with inline comments explaining key design choices.
