import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ==========================================
# 1. Synthetic Data Generation (The "Hidden" Gap)
# ==========================================
def generate_hhw_data(n_samples=2000):
    """
    Simulates a dataset of Household Hazardous Waste items with features 
    relevant to treatment decisions.
    """
    np.random.seed(42)
    
    # Feature 1: Waste Category
    categories = ['Battery', 'Paint', 'Cleaner', 'Pesticide', 'E-Waste', 'Oil']
    waste_types = np.random.choice(categories, n_samples)
    
    # Feature 2: pH Level (0-14) - Critical for chemical treatment
    ph_levels = []
    for w in waste_types:
        if w == 'Cleaner': ph_levels.append(np.random.uniform(11, 14)) # Basic
        elif w == 'Battery': ph_levels.append(np.random.uniform(0, 3))   # Acidic
        else: ph_levels.append(np.random.uniform(6, 8))                  # Neutral
    
    # Feature 3: Flammability Score (0-10)
    flammability = []
    for w in waste_types:
        if w in ['Paint', 'Oil']: flammability.append(np.random.uniform(7, 10))
        else: flammability.append(np.random.uniform(0, 3))
        
    # Feature 4: Container Integrity (0=Leaking, 1=Intact)
    integrity = np.random.choice([0, 1], n_samples, p=[0.2, 0.8])
    
    df = pd.DataFrame({
        'Waste_Type': waste_types,
        'pH_Level': np.round(ph_levels, 1),
        'Flammability': np.round(flammability, 1),
        'Container_Intact': integrity
    })

    # Logic for "Ground Truth" Treatment (The Target Variable)
    # This simulates expert labeling rules
    treatments = []
    for i, row in df.iterrows():
        if row['Waste_Type'] == 'E-Waste':
            treatments.append('Recycling (E-Specialist)')
        elif row['Waste_Type'] == 'Battery':
            treatments.append('Neutralization & Metal Recovery')
        elif row['Flammability'] > 8:
            treatments.append('High-Temp Incineration')
        elif row['pH_Level'] > 12 or row['pH_Level'] < 2:
            treatments.append('Chemical Neutralization')
        else:
            treatments.append('Secure Landfill')
            
    df['Recommended_Treatment'] = treatments
    return df

# ==========================================
# 2. Preprocessing & Model Training
# ==========================================
print("Generating Synthetic HHW Data...")
df = generate_hhw_data()

# Encode Categorical 'Waste_Type' for the model
le_waste = LabelEncoder()
df['Waste_Type_Encoded'] = le_waste.fit_transform(df['Waste_Type'])

features = ['Waste_Type_Encoded', 'pH_Level', 'Flammability', 'Container_Intact']
X = df[features]
y = df['Recommended_Treatment']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model (Random Forest is used here for stability, XGBoost is drop-in replaceable)
print("Training X-IDSS Model...")
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Evaluate
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {acc*100:.2f}%")

# ==========================================
# 3. Explainability Module (SHAP)
# ==========================================
print("Initializing SHAP Explainer...")
explainer = shap.TreeExplainer(model)
# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test)

def predict_and_explain(waste_type, ph, flam, intact):
    """
    Simulates the User Interface: Input -> Prediction -> Explanation
    """
    # Encode input
    type_code = le_waste.transform([waste_type])[0]
    input_data = pd.DataFrame([[type_code, ph, flam, intact]], columns=features)
    
    # 1. Prediction
    prediction = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)
    
    # 2. Explanation logic
    shap_vals_all = explainer.shap_values(input_data)
    class_idx = np.where(model.classes_ == prediction)[0][0]
    
    # --- ROBUST SHAPE HANDLING ---
    if isinstance(shap_vals_all, list):
        # Case A: SHAP returns a list of arrays (one per class)
        raw_vals = shap_vals_all[class_idx]
    else:
        # Case B: SHAP returns a single array (3D or 2D)
        if len(shap_vals_all.shape) == 3:
            raw_vals = shap_vals_all[:, :, class_idx]
        else:
            raw_vals = shap_vals_all

    # Force it to be a simple 1D list of numbers
    # .flatten() ensures we don't have shapes like (1, 4) or (4, 1)
    shap_val_single = np.array(raw_vals).flatten()
    
    # Truncate if it somehow got extra data (safety check)
    if len(shap_val_single) > len(features):
        shap_val_single = shap_val_single[:len(features)]
    # -----------------------------

    print("\n" + "="*40)
    print(f"USER INPUT: {waste_type} | pH: {ph} | Flam: {flam}")
    print("="*40)
    print(f">>> SYSTEM RECOMMENDATION: {prediction}")
    print(f"    (Confidence: {np.max(probs)*100:.1f}%)")
    print("-" * 40)
    print(">>> WHY THIS DECISION? (XAI Explanation)")
    
    feature_names = ['Waste Type', 'pH Level', 'Flammability', 'Container State']
    
    for name, val in zip(feature_names, shap_val_single):
        # val is now guaranteed to be a simple float
        if val > 0:
            impact = "Increases Risk/Likelihood"
        else:
            impact = "Decreases Risk/Likelihood"
            
        # Only print if the impact is significant (absolute value > 0.01)
        if abs(val) > 0.01: 
            print(f"  * {name}: {impact} (Impact Score: {val:.3f})")
    print("="*40 + "\n")

# ==========================================
# 4. Interactive Demo Scenarios
# ==========================================

# Scenario A: Old Paint Thinner
predict_and_explain(waste_type='Paint', ph=7.0, flam=9.5, intact=1)

# Scenario B: Leaking Car Battery
predict_and_explain(waste_type='Battery', ph=1.0, flam=2.0, intact=0)

# Optional: Visualization (Uncomment to see plot)
# shap.summary_plot(shap_values, X_test, feature_names=features)