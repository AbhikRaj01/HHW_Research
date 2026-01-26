import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# =========================================================
# 1. REALISTIC LIQUID HAZARDOUS HOUSEHOLD WASTE DATA
# =========================================================
def generate_lhhw_data(n_samples=2500):
    np.random.seed(42)

    liquid_categories = [
        'Acid', 'Alkali', 'Solvent', 'Oil', 'Pesticide', 'Liquid_Paint',
        'Industrial_Cleaner', 'Pharmaceutical_Liquid', 'Automotive_Fluid',
        'Laboratory_Chemical', 'Fuel_Residue', 'Disinfectant'
    ]

    waste_types = np.random.choice(liquid_categories, n_samples)

    ph_levels, flammability, toxicity, reactivity, corrosiveness, viscosity = [], [], [], [], [], []

    for w in waste_types:
        p_val = np.random.normal(7, 2)
        f_val = np.random.normal(3, 2)
        t_val = np.random.normal(4, 2)
        r_val = np.random.normal(3, 2)
        c_val = np.random.normal(3, 2)
        v_val = np.random.normal(5, 2)

        if w == 'Acid':
            p_val -= np.random.uniform(3, 6)
            c_val += np.random.uniform(4, 7)
        elif w == 'Alkali':
            p_val += np.random.uniform(3, 6)
            c_val += np.random.uniform(4, 7)
        elif w in ['Solvent', 'Fuel_Residue', 'Liquid_Paint']:
            f_val += np.random.uniform(4, 6)
        elif w == 'Pesticide':
            t_val += np.random.uniform(4, 6)
        elif w == 'Laboratory_Chemical':
            r_val += np.random.uniform(4, 6)
            t_val += np.random.uniform(3, 5)
        elif w == 'Oil':
            v_val += np.random.uniform(3, 5)
        elif w == 'Industrial_Cleaner':
            p_val += np.random.uniform(2, 4)
            t_val += np.random.uniform(2, 4)

        ph_levels.append(np.clip(p_val, 0, 14))
        flammability.append(np.clip(f_val, 0, 10))
        toxicity.append(np.clip(t_val, 0, 10))
        reactivity.append(np.clip(r_val, 0, 10))
        corrosiveness.append(np.clip(c_val, 0, 10))
        viscosity.append(np.clip(v_val, 0, 10))

    integrity = np.random.choice([0, 1], n_samples, p=[0.25, 0.75])

    df = pd.DataFrame({
        'Liquid_Type': waste_types,
        'pH_Level': np.round(ph_levels, 1),
        'Flammability': np.round(flammability, 1),
        'Toxicity_Score': np.round(toxicity, 1),
        'Reactivity': np.round(reactivity, 1),
        'Corrosiveness': np.round(corrosiveness, 1),
        'Viscosity': np.round(viscosity, 1),
        'Container_Intact': integrity
    })

    # =====================================================
    # REALISTIC LIQUID WASTE TREATMENT LOGIC
    # =====================================================
    treatments = []
    for _, row in df.iterrows():
        ph = row['pH_Level']
        tox = row['Toxicity_Score']
        flam = row['Flammability']
        react = row['Reactivity']
        corr = row['Corrosiveness']
        wtype = row['Liquid_Type']

        if tox > 8 and react > 6:
            treatments.append('Advanced Hazardous Chemical Treatment Facility')
        elif ph < 3 or ph > 11:
            treatments.append('Chemical Neutralization')
        elif flam > 7 and wtype in ['Solvent', 'Fuel_Residue', 'Liquid_Paint']:
            treatments.append('Solvent Recovery & Controlled Incineration')
        elif wtype in ['Oil', 'Automotive_Fluid']:
            treatments.append('Oil Recovery & Re-refining')
        elif wtype == 'Pesticide':
            treatments.append('Secure Toxic Chemical Disposal')
        elif react > 7:
            treatments.append('Chemical Stabilization')
        elif tox > 6:
            treatments.append('Hazardous Liquid Treatment Plant')
        else:
            treatments.append('Secure Liquid Waste Landfill')

    df['Recommended_Treatment'] = treatments
    return df


# =========================================================
# 2. DATASET + MODEL TRAINING
# =========================================================
print("Generating Liquid HHW Dataset...")
df = generate_lhhw_data()

le_waste = LabelEncoder()
df['Liquid_Type_Encoded'] = le_waste.fit_transform(df['Liquid_Type'])

features = ['Liquid_Type_Encoded', 'pH_Level', 'Flammability', 'Toxicity_Score', 'Reactivity', 'Corrosiveness', 'Viscosity', 'Container_Intact']
X = df[features]
y = df['Recommended_Treatment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("Training Machine Learning Model...")
model = RandomForestClassifier(n_estimators=180, max_depth=9, random_state=42)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {acc*100:.2f}%")

print("Initializing SHAP...")
explainer = shap.TreeExplainer(model)

# =========================================================
# 3. HAZARD SCORE + RISK CLASSIFICATION
# =========================================================
def calculate_hazard_score(ph, flam, tox, react, corr, visc, intact):
    score = (
        tox * 3 +
        flam * 2 +
        react * 2 +
        corr * 2 +
        visc * 1.2 +
        abs(ph - 7) +
        (6 if intact == 0 else 0)
    )
    return min(round(score * 1.8, 2), 100)

def classify_risk(score):
    if score < 25:
        return "LOW RISK"
    elif score < 50:
        return "MODERATE RISK"
    elif score < 75:
        return "HIGH RISK"
    else:
        return "CRITICAL RISK"


# =========================================================
# 4. SAFE SHAP HANDLER (FIXED)
# =========================================================
def get_shap_values(input_data, prediction):
    shap_vals_all = explainer.shap_values(input_data)

    # Case 1: multiclass -> list
    if isinstance(shap_vals_all, list):
        class_idx = np.where(model.classes_ == prediction)[0][0]
        shap_vals = shap_vals_all[class_idx]

    # Case 2: ndarray (binary or sklearn version change)
    else:
        if len(shap_vals_all.shape) == 3:
            class_idx = np.where(model.classes_ == prediction)[0][0]
            shap_vals = shap_vals_all[:, :, class_idx]
        else:
            shap_vals = shap_vals_all

    return np.array(shap_vals).flatten()


# =========================================================
# 5. GRAPH MENU SYSTEM
# =========================================================
def graph_menu(name, values):
    while True:
        print("\n--- Visualization Menu ---")
        print("1. Hazard Profile Bar Graph")
        print("2. Radar Risk Chart")
        print("3. Feature Correlation Heatmap (Global)")
        print("4. SHAP Summary Plot (Global)")
        print("5. Back")

        choice = input("Select option (1-5): ")

        if choice == '1':
            plt.figure()
            labels = ['pH','Flammability','Toxicity','Reactivity','Corrosiveness','Viscosity']
            plt.bar(labels, values)
            plt.title(f"Hazard Profile: {name}")
            plt.show()

        elif choice == '2':
            labels = ['pH','Flammability','Toxicity','Reactivity','Corrosiveness','Viscosity']
            angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
            values_radar = np.concatenate((values, [values[0]]))
            angles = np.concatenate((angles, [angles[0]]))

            plt.figure()
            plt.polar(angles, values_radar)
            plt.fill(angles, values_radar, alpha=0.3)
            plt.title(f"Radar Risk Profile: {name}")
            plt.show()

        elif choice == '3':
            plt.figure()
            sns.heatmap(df[['pH_Level','Flammability','Toxicity_Score','Reactivity','Corrosiveness','Viscosity']].corr(), annot=True, cmap='coolwarm')
            plt.title("Feature Correlation Heatmap (LHHW Dataset)")
            plt.show()

        elif choice == '4':
            shap_values_all = explainer.shap_values(X_test, check_additivity=False)
            shap_matrix = shap_values_all[0] if isinstance(shap_values_all, list) else shap_values_all
            shap.summary_plot(shap_matrix, X_test, feature_names=features, show=True)

        elif choice == '5':
            break


# =========================================================
# 6. PREDICTION + EXPLANATION ENGINE
# =========================================================
scenario_reports = []

def predict_and_explain(name, liquid_type, ph, flam, tox, react, corr, visc, intact):
    type_code = le_waste.transform([liquid_type])[0]

    input_data = pd.DataFrame([[type_code, ph, flam, tox, react, corr, visc, intact]], columns=features)

    prediction = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)

    shap_vals = get_shap_values(input_data, prediction)

    hazard_score = calculate_hazard_score(ph, flam, tox, react, corr, visc, intact)
    risk_level = classify_risk(hazard_score)

    print("\n" + "="*90)
    print(f"LHHW SCENARIO: {name}")
    print(f"Liquid Type: {liquid_type}")
    print("="*90)
    print(f">>> RECOMMENDED TREATMENT: {prediction}")
    print(f">>> CONFIDENCE: {np.max(probs)*100:.2f}%")
    print(f">>> HAZARD SCORE: {hazard_score}/100")
    print(f">>> RISK LEVEL: {risk_level}")
    print("-"*90)
    print("AI EXPLANATION (SHAP FEATURE IMPACT):")

    feature_names_disp = ['Liquid Type','pH Level','Flammability','Toxicity','Reactivity','Corrosiveness','Viscosity','Container State']
    zipped = sorted(zip(feature_names_disp, shap_vals[:len(feature_names_disp)]), key=lambda x: abs(x[1]), reverse=True)

    for name_f, val in zipped:
        impact = "Increases Hazard" if val > 0 else "Reduces Hazard"
        if abs(val) > 0.01:
            print(f"  * {name_f:<15}: {impact} (Score: {val:.3f})")

    print("="*90)

    values = [ph, flam, tox, react, corr, visc]
    graph_menu(name, values)

    scenario_reports.append({
        "Scenario": name,
        "Liquid_Type": liquid_type,
        "Treatment": prediction,
        "Hazard_Score": hazard_score,
        "Risk_Level": risk_level
    })


# =========================================================
# 7. REALISTIC LIQUID SCENARIOS
# =========================================================
SCENARIOS = [
    ("Sulfuric Acid Spill", 'Acid', 1.5, 1.0, 7.5, 4.0, 8.5, 3.0, 0),
    ("Caustic Soda Solution", 'Alkali', 13.2, 0.5, 6.5, 3.0, 7.5, 3.0, 0),
    ("Paint Thinner", 'Solvent', 6.8, 9.0, 5.5, 3.0, 3.0, 2.0, 1),
    ("Engine Oil Waste", 'Oil', 6.5, 7.5, 4.5, 2.0, 2.0, 8.0, 1),
    ("Agricultural Pesticide", 'Pesticide', 6.2, 2.5, 9.0, 4.0, 3.0, 3.0, 0),
    ("Liquid Wall Paint", 'Liquid_Paint', 7.0, 8.5, 4.0, 2.0, 2.0, 5.0, 1),
    ("Industrial Cleaner", 'Industrial_Cleaner', 12.5, 1.5, 6.0, 2.0, 5.5, 3.0, 1),
    ("Expired Syrup Medicine", 'Pharmaceutical_Liquid', 7.0, 1.0, 5.5, 2.0, 2.0, 4.0, 1),
    ("Brake Fluid", 'Automotive_Fluid', 6.8, 6.5, 5.5, 3.0, 3.0, 6.0, 1),
    ("Lab Chemical Mix", 'Laboratory_Chemical', 6.0, 4.5, 7.5, 5.5, 4.0, 3.0, 0),
]


# =========================================================
# 8. INTERACTIVE MENU
# =========================================================
def run_menu():
    while True:
        print("\n========== LIQUID HAZARDOUS HOUSEHOLD WASTE AI SYSTEM ==========\n")
        for i, sc in enumerate(SCENARIOS):
            print(f"{i+1}. {sc[0]}")
        print(f"{len(SCENARIOS)+1}. Custom Liquid Input")
        print(f"{len(SCENARIOS)+2}. Exit")

        choice = int(input(f"\nEnter choice (1-{len(SCENARIOS)+2}): "))

        if 1 <= choice <= len(SCENARIOS):
            predict_and_explain(*SCENARIOS[choice-1])
        elif choice == len(SCENARIOS)+1:
            name = "Custom Liquid Scenario"
            w = input("Liquid Type: ")
            p = float(input("pH (0-14): "))
            f = float(input("Flammability (0-10): "))
            t = float(input("Toxicity (0-10): "))
            r = float(input("Reactivity (0-10): "))
            c = float(input("Corrosiveness (0-10): "))
            v = float(input("Viscosity (0-10): "))
            i_val = int(input("Intact (1=Yes, 0=No): "))
            predict_and_explain(name, w, p, f, t, r, c, v, i_val)
        elif choice == len(SCENARIOS)+2:
            break


# =========================================================
# 9. FINAL REPORT
# =========================================================
if __name__ == "__main__":
    run_menu()

    print("\n================ FINAL LHHW AI REPORT ================\n")
    report_df = pd.DataFrame(scenario_reports)
    print(report_df)

    plt.figure()
    report_df['Risk_Level'].value_counts().plot(kind='bar')
    plt.title("Risk Level Distribution Across Liquid Waste Scenarios")
    plt.show()