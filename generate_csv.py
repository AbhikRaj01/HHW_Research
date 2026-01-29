import pandas as pd
import numpy as np

# ==========================================
# DATA GENERATION FUNCTION
# ==========================================
def generate_lhhw_csv(n_samples=200):
    np.random.seed(42)  # Ensures the same 200 rows every time you run it

    liquid_categories = [
        'Acid', 'Alkali', 'Solvent', 'Oil', 'Pesticide', 'Liquid_Paint',
        'Industrial_Cleaner', 'Pharmaceutical_Liquid', 'Automotive_Fluid',
        'Laboratory_Chemical', 'Fuel_Residue', 'Disinfectant'
    ]

    waste_types = np.random.choice(liquid_categories, n_samples)

    ph_levels, flammability, toxicity, reactivity, corrosiveness, viscosity = [], [], [], [], [], []

    for w in waste_types:
        # 1. Base Values (Normal Distribution)
        p_val = np.random.normal(7, 2)
        f_val = np.random.normal(3, 2)
        t_val = np.random.normal(4, 2)
        r_val = np.random.normal(3, 2)
        c_val = np.random.normal(3, 2)
        v_val = np.random.normal(5, 2)

        # 2. Apply Chemical Rules (The "Logic")
        if w == 'Acid':
            p_val -= np.random.uniform(3, 6) # Low pH
            c_val += np.random.uniform(4, 7) # High Corrosiveness
        elif w == 'Alkali':
            p_val += np.random.uniform(3, 6) # High pH
            c_val += np.random.uniform(4, 7)
        elif w in ['Solvent', 'Fuel_Residue', 'Liquid_Paint']:
            f_val += np.random.uniform(4, 6) # High Flammability
        elif w == 'Pesticide':
            t_val += np.random.uniform(4, 6) # High Toxicity
        elif w == 'Laboratory_Chemical':
            r_val += np.random.uniform(4, 6) # High Reactivity
            t_val += np.random.uniform(3, 5)
        elif w == 'Oil':
            v_val += np.random.uniform(3, 5) # High Viscosity
        elif w == 'Industrial_Cleaner':
            p_val += np.random.uniform(2, 4)
            t_val += np.random.uniform(2, 4)

        # 3. Clip values to stay in realistic ranges
        ph_levels.append(np.clip(p_val, 0, 14))
        flammability.append(np.clip(f_val, 0, 10))
        toxicity.append(np.clip(t_val, 0, 10))
        reactivity.append(np.clip(r_val, 0, 10))
        corrosiveness.append(np.clip(c_val, 0, 10))
        viscosity.append(np.clip(v_val, 0, 10))

    # 4. Container Integrity (0=Leaking, 1=Intact)
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

    # 5. Assign Treatment Labels (Ground Truth)
    treatments = []
    for _, row in df.iterrows():
        ph = row['pH_Level']
        tox = row['Toxicity_Score']
        flam = row['Flammability']
        react = row['Reactivity']
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

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Generate 200 rows
    df_200 = generate_lhhw_csv(n_samples=200)
    
    # Save to CSV
    df_200.to_csv('lhhw_training_data.csv', index=False)
    
    print("Success! Generated 'lhhw_training_data.csv' with 200 rows.")
    print(df_200.head(10)) # Show first 10 rows