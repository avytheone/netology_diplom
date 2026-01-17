# AGENTS.md

This file contains guidelines and commands for agentic coding agents working in this diabetes analysis project.

## Project Overview

This is a Python data science project focused on cardiovascular disease analysis using:
- Python 3.12+ 
- UV package manager
- Scientific libraries: pandas, numpy, matplotlib, polars
- Quarto for documentation generation
- Research-oriented project structure

## Environment Setup

The project uses UV for dependency management. The virtual environment is located in `.venv/`.

```bash
# Install dependencies
uv sync

# Run commands in the project environment
uv run <command>

# Activate the environment (if needed)
source .venv/bin/activate
```

## Build and Documentation Commands

```bash
# Build Quarto documentation
uv run quarto render

# Build specific file
uv run quarto render research.qmd

# Build presentation
uv run quarto render presentation.qmd

# Serve documentation for development
uv run quarto preview --port 4200

# Build all formats
uv run quarto render --to html,pdf
```

## Code Execution Commands

```bash
# Run Python scripts
uv run python script.py

# Run Jupyter notebooks (if available)
uv run jupyter notebook

# Run specific Python module
uv run python -m module_name
```

## Testing Commands

This project does not currently have formal tests. For data validation:

```bash
# Run data validation scripts
uv run python validate_data.py

# Check data integrity
uv run python -c "import pandas as pd; print(pd.read_csv('data.csv').info())"
```

## Code Style Guidelines

### Python Code Style

Follow PEP 8 with these project-specific conventions:

#### Import Organization
```python
# Standard library imports first
import os
import sys
from pathlib import Path

# Third-party imports next
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import polars as pl

# Local imports last (if any)
from utils import helper_function
```

#### Naming Conventions
- Variables: `snake_case` (e.g., `patient_data`, `blood_pressure`)
- Functions: `snake_case` with descriptive names (e.g., `load_cardio_data`, `calculate_risk_score`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DATA_PATH`, `MAX_AGE`)
- Classes: `PascalCase` (e.g., `DataProcessor`, `RiskModel`)

#### Type Hints
Use type hints for all function signatures and important variables:

```python
from typing import Optional, Union, List, Dict, Tuple
import pandas as pd

def process_patient_data(
    data: pd.DataFrame, 
    age_column: str = 'age',
    drop_missing: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Process patient data and return cleaned dataframe with statistics."""
    pass
```

#### Error Handling
Use specific exception handling with logging:

```python
import logging

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """Load cardiovascular dataset."""
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data with shape {data.shape}")
        return data
    except FileNotFoundError:
        logger.error(f"Data file not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error("Data file is empty")
        raise ValueError("Empty data file")
```

### Data Analysis Patterns

#### Data Loading
```python
def load_cardiovascular_data(path: str) -> pd.DataFrame:
    """Load and validate cardiovascular disease dataset."""
    df = pd.read_csv(path)
    
    # Basic validation
    required_columns = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df
```

#### Visualization Style
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set consistent style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_correlation_heatmap(data: pd.DataFrame, title: str = "Feature Correlation") -> plt.Figure:
    """Create correlation heatmap with consistent styling."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title(title)
    return fig
```

## File Structure

```
diabetes/
├── pyproject.toml          # Project configuration
├── uv.lock                 # Dependency lock file
├── README.md               # Project description
├── research.qmd           # Main research document
├── presentation.qmd       # Presentation slides
├── docs/                   # Additional documentation
├── out/                    # Generated outputs
├── fonts/                  # Custom fonts (if any)
└── .venv/                  # Virtual environment
```

## Data Guidelines

### Data Sources
- Primary dataset: Cardiovascular Disease Dataset from Kaggle
- Expected format: CSV with standardized column names

### Data Processing Pipeline
1. Load data with validation
2. Clean missing values and outliers
3. Perform exploratory data analysis
4. Create visualizations
5. Build predictive models (if applicable)
6. Generate insights and recommendations

### Model Building (if applicable)
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def build_risk_model(data: pd.DataFrame, target: str = 'cardio') -> Dict[str, Any]:
    """Build logistic regression model for cardiovascular risk prediction."""
    # Prepare features
    X = data.drop(columns=[target])
    y = data[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
```

## Documentation Standards

### Quarto Documents
- Use YAML front matter with proper metadata
- Include code chunks with meaningful labels
- Add figure captions and table descriptions
- Use Russian language for main content (as per project context)

### Code Comments
- Add docstrings for all functions and classes
- Use inline comments for complex logic
- Include TODO comments for future improvements

## Best Practices

1. **Reproducibility**: Set random seeds for all stochastic processes
2. **Performance**: Use polars for large datasets, pandas for smaller ones
3. **Visualization**: Create publication-ready plots with proper labels and legends
4. **Documentation**: Keep README and documentation updated with project progress
5. **Version Control**: Commit frequently with descriptive messages

## Common Issues and Solutions

### Memory Issues
```python
# For large datasets, use polars instead of pandas
import polars as pl
df = pl.read_csv('large_dataset.csv')
```

### Rendering Issues
```bash
# If Quarto rendering fails, check dependencies
uv run quarto check

# Reinstall if needed
uv add --dev quarto
```

### Data Validation
```python
# Always validate data after loading
def validate_data(df: pd.DataFrame) -> bool:
    """Basic data validation checks."""
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if df.isnull().all().all():
        raise ValueError("DataFrame contains only null values")
    
    return True
```

## Language and Localization

- Project documentation is primarily in Russian
- Code comments and variable names should be in English
- Error messages can be in Russian for user-facing components
- Follow Russian academic formatting standards for final output