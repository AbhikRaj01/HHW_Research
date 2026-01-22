import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ==========================================
# 1. EXPANDED Synthetic Data Generation
# ==========================================
def generate_complex_hhw_data(n_samples=3000):
    np.random.seed(42)
    categories = [
        'Battery', 'Paint', 'Cleaner', 'Pesticide', 'E-Waste', 
        'Oil', 'Fluorescent_Bulb', 'Medical_Waste', 'Aerosol'
    ]
    waste_types = np.random.choice(categories, n_samples)
    
    ph_levels = []
    flammability = []
    toxicity = [] 
    
    for w in waste_types:
        p_val = np.random.uniform(6, 8) 
        f_val = np.random.uniform(0, 2) 
        t_val = np.random.uniform(0, 3) 
        
        if w == 'Cleaner': 
            p_val = np.random.uniform(10, 14) 
            t_val = np.random.uniform(4, 7)
        elif w == 'Battery': 
            p_val = np.random.uniform(0, 4)   
            t_val = np.random.uniform(5, 9)
        elif w in ['Paint', 'Oil', 'Aerosol']: 
            f_val = np.random.uniform(7, 10)  
            t_val = np.random.uniform(3, 6)
        elif w == 'Pesticide':
            t_val = np.random.uniform(8, 10)  
        elif w == 'Fluorescent_Bulb':
            t_val = np.random.uniform(6, 9)   
        elif w == 'Medical_Waste':
            t_val = np.random.uniform(5, 8)   
            
        ph_levels.append(p_val)
        flammability.append(f_val)
        toxicity.append(t_val)
        
    integrity = np.random.choice([0, 1], n_samples, p=[0.15, 0.85])
    
    df = pd.DataFrame({
        'Waste_Type': waste_types,
        'pH_Level': np.round(ph_levels, 1),
        'Flammability': np.round(flammability, 1),
        'Toxicity_Score': np.round(toxicity, 1),
        'Container_Intact': integrity
    })

    treatments = []
    for i, row in df.iterrows():
        if row['Waste_Type'] == 'Medical_Waste':
             treatments.append('Bio-Incineration / Autoclave')
        elif row['Waste_Type'] == 'Fluorescent_Bulb':
             treatments.append('Mercury Retort (Specialized Recycling)')
        elif row['Waste_Type'] == 'Aerosol' and row['Flammability'] > 5:
             treatments.append('Depressurization & Incineration')
        elif row['Waste_Type'] == 'E-Waste':
            treatments.append('E-Waste Dismantling & Recovery')
        elif row['Waste_Type'] == 'Battery':
            treatments.append('Battery Neutralization & Smelting')
        elif row['Flammability'] > 8:
            treatments.append('High-Temp Incineration')
        elif row['Toxicity_Score'] > 8:
             treatments.append('Hazardous Waste Landfill (Stabilized)')
        elif row['pH_Level'] > 12 or row['pH_Level'] < 2:
            treatments.append('Chemical Neutralization')
        else:
            treatments.append('Standard Secure Landfill')
            
    df['Recommended_Treatment'] = treatments
    return df

# ==========================================
# 2. Training
# ==========================================
print("Generating Expanded Dataset...")
df = generate_complex_hhw_data()

le_waste = LabelEncoder()
df['Waste_Type_Encoded'] = le_waste.fit_transform(df['Waste_Type'])

features = ['Waste_Type_Encoded', 'pH_Level', 'Flammability', 'Toxicity_Score', 'Container_Intact']
X = df[features]
y = df['Recommended_Treatment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Advanced Model...")
model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {acc*100:.2f}%")

print("Initializing SHAP...")
explainer = shap.TreeExplainer(model)

# ==========================================
# 3. Explanation Function
# ==========================================
def predict_and_explain(waste_type, ph, flam, tox, intact):
    try:
        type_code = le_waste.transform([waste_type])[0]
    except ValueError:
        print(f"Error: '{waste_type}' is not a known category.")
        return

    input_data = pd.DataFrame([[type_code, ph, flam, tox, intact]], columns=features)
    
    prediction = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)
    
    shap_vals_all = explainer.shap_values(input_data)
    class_idx = np.where(model.classes_ == prediction)[0][0]
    
    if isinstance(shap_vals_all, list):
        raw_vals = shap_vals_all[class_idx]
    else:
        # Handle 3D or 2D arrays
        if len(shap_vals_all.shape) == 3:
            raw_vals = shap_vals_all[:, :, class_idx]
        else:
            raw_vals = shap_vals_all

    shap_val_single = np.array(raw_vals).flatten()
    if len(shap_val_single) > len(features):
        shap_val_single = shap_val_single[:len(features)]

    print("\n" + "="*50)
    print(f" SCENARIO: {waste_type} | pH: {ph} | Flam: {flam} | Tox: {tox}")
    print("="*50)
    print(f">>> RECOMMENDATION: {prediction}")
    print(f"    (Confidence: {np.max(probs)*100:.1f}%)")
    print("-" * 50)
    print(">>> EXPLANATION (Key Drivers):")
    
    feature_names_disp = ['Waste Type', 'pH Level', 'Flammability', 'Toxicity', 'Container State']
    zipped = sorted(zip(feature_names_disp, shap_val_single), key=lambda x: abs(x[1]), reverse=True)

    for name, val in zipped:
        impact = "Increases Risk" if val > 0 else "Decreases Risk"
        if abs(val) > 0.01: 
            print(f"  * {name:<15}: {impact} (Score: {val:.3f})")
    print("="*50 + "\n")

# ==========================================
# 4. Interactive Menu
# ==========================================
def run_menu():
    while True:
        print("\nSelect a Test Scenario:")
        print("1. Broken Fluorescent Bulb (Mercury Risk)")
        print("2. Expired Medicine / Biohazard")
        print("3. Highly Flammable Spray Can")
        print("4. Super Toxic Pesticide")
        print("5. Custom Input")
        print("6. Exit")
        
        choice = input("Enter choice (1-6): ")
        
        if choice == '1':
            predict_and_explain('Fluorescent_Bulb', ph=7.0, flam=0.0, tox=8.5, intact=0)
        elif choice == '2':
            predict_and_explain('Medical_Waste', ph=7.0, flam=0.0, tox=6.5, intact=1)
        elif choice == '3':
            predict_and_explain('Aerosol', ph=7.0, flam=9.8, tox=4.0, intact=1)
        elif choice == '4':
            predict_and_explain('Pesticide', ph=6.5, flam=2.0, tox=9.5, intact=0)
        elif choice == '5':
            print("\nEnter details: [Battery, Paint, Cleaner, Pesticide, E-Waste, Oil, Fluorescent_Bulb, Medical_Waste, Aerosol]")
            w = input("Waste Type: ")
            p = float(input("pH (0-14): "))
            f = float(input("Flammability (0-10): "))
            t = float(input("Toxicity (0-10): "))
            i = int(input("Intact (1=Yes, 0=No): "))
            predict_and_explain(w, p, f, t, i)
        elif choice == '6':
            print("Exiting and Generating Graph...")
            break
# ==========================================
# 5. VISUALIZATION BLOCK (ABSOLUTE FIX)
# ==========================================
if __name__ == "__main__":
    run_menu()

    print("Calculating SHAP values for summary plot...")
    shap_values_all = explainer.shap_values(X_test, check_additivity=False)

    # --- HANDLE ALL SHAP OUTPUT FORMATS ---
    if isinstance(shap_values_all, list):
        # Case 1: Proper multiclass list
        shap_matrix = shap_values_all[0]

        class_name = model.classes_[0]
        print(f"\n[INFO] Displaying Feature Importance for Class: '{class_name}'")

    elif isinstance(shap_values_all, np.ndarray) and shap_values_all.ndim == 3:
        # Case 2: 3D array → select one class explicitly
        shap_matrix = shap_values_all[:, :, 0]

        class_name = model.classes_[0]
        print(f"\n[INFO] Displaying Feature Importance for Class: '{class_name}'")

    else:
        # Case 3: Binary or already-correct shape
        shap_matrix = shap_values_all

        print("\n[INFO] Displaying Feature Importance (Binary Mode)")

    # --- FINAL SAFE PLOT ---
    shap.summary_plot(
        shap_matrix,
        X_test,
        feature_names=features,
        show=False
    )

    plt.tight_layout()
    plt.show()
