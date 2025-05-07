# Healthylife-Insurance-Charge-Prediction

🚀 Project Overview
Business Context
Healthylife is a leading insurance provider based in New York City offering health, auto, and life policies nationwide. Today, charges are set via traditional rate tables based on age, sex, BMI, etc., but these lack precision—leading to under‐ or over‐pricing that can hurt both profits and customer satisfaction.

Problem Statement
- Accurately predict individual insurance charges given customer attributes
- Understand how factors like age, BMI, smoking status, and region drive costs
- Streamline underwriting with a data‐driven model, while ensuring regulatory compliance

Objective
Build and deploy a predictive model (and accompanying web API/app) that:
1. Ingests customer data (age, sex, BMI, children, smoker, region)
2. Outputs a tailored insurance charge estimate
3. Monitors for data/model drift over time

---

📂 Repository Structure
```pgsql
README.md
requirements.txt               ← Python dependencies
insurance_charge_model.pkl     ← Trained model artifact
notebooks/
  └─ insurance_charge_prediction.ipynb  ← Jupyter notebook with full EDA & modelling
src/
  ├─ data_preparation.py       ← Scripts to load & preprocess data  
  ├─ train_model.py            ← Script to train & serialize the model  
  └─ predict.py                ← Script/API for loading the model and making predictions  
data/
  └─ insurance.csv             ← Raw dataset (6 183 rows × 7 cols)  
```

---

🛠️ Setup & Dependencies
1. Python: tested on 3.8+
2. Install requirements:
```bash
pip install -r requirements.txt
```
3. Dataset
  - Download or place `insurance.csv` in the `data/` folder

---

📥 Data Import & Preprocessing
In your notebook or script:
```python
import pandas as pd

df = pd.read_csv("data/insurance.csv")
print(df.shape)    # (6183, 7)
df.head()
```

- Numeric features: `age`, `bmi`, `children`
- Categorical features: `sex`, `smoker`, `region`
- Target: `charges` 

Preprocessing pipeline (scikit‐learn `ColumnTransformer`):
- Numeric: `SimpleImputer(strategy='median')` → `StandardScaler()`
- Categorical: `OneHotEncoder(handle_unknown='ignore')`

---

🔍 Exploratory Data Analysis (EDA)
Key insights:
- Age and BMI show strong positive correlation with charges.
- Smokers incur on average ~10× higher charges than non‐smokers.
- Regional differences exist but are smaller than the smoker effect.
(See full EDA in *`notebooks/insurance_charge_prediction.ipynb`*.)

---

📊 Modeling
Model choice:
- Simple Linear Regression with all preprocessing in a single `Pipeline` 
- (You can swap in more advanced regressors - e.g. RandomForestRegressor, XGBoost, etc.)

Training:
```bash
python src/train_model.py \
  --data-path data/insurance.csv \
  --model-out insurance_charge_model.pkl
```

Evaluation metrics:
- RMSE, MAE, and R² on hold‐out test set
- Residual analysis to validate assumptions
(See “Model Evaluation” section in the notebook.) 

---

🚧 Usage
Load the trained model and predict on new data:
```python
from src.predict import load_model, predict_charges

model = load_model("insurance_charge_model.pkl")

new_customers = [
    {"age": 45, "sex": "male", "bmi": 32.5, "children": 2, "smoker": "yes", "region": "southeast"},
    # …
]
predictions = predict_charges(model, new_customers)
print(predictions)
```

Or via command‐line:
```bash
python src/predict.py \
  --model insurance_charge_model.pkl \
  --input new_customers.json \
  --output predictions.csv
```

---

🤝 Contributing
- Issues & feature requests are welcome.
- For code contributions, please fork, create a feature branch, and submit a pull request.










