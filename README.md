# ♻️ X-IDSS: Explainable AI for Hazardous Waste

> **An Intelligent Decision Support System (IDSS) for Sustainable Household Hazardous Waste (HHW) Treatment.**

## 📖 Overview

**Household Hazardous Waste (HHW)**—like lithium batteries, paints, and corrosive cleaners—is often disposed of incorrectly because the average person lacks the expertise to distinguish between safe and dangerous items.

This project bridges the **"Decision Gap"**. Unlike traditional "Black Box" AI that merely sorts waste, this system is an **X-IDSS (Explainable Intelligent Decision Support System)**. It not only recommends the safest treatment method (e.g., Neutralization, Resource Recovery) but also **explains why**, building trust and ensuring safety.

## ⚠️ The Problem Landscape

Why do we need this?

1. **Complexity:** Users can't distinguish between a standard zinc-carbon battery (landfill-safe) and a lithium-polymer battery (fire risk).


2. **Consequence:** Improper disposal leads to "thermal runaway" fires in garbage trucks and groundwater contamination.


3. **The "Black Box" Issue:** Existing AI can identify an object ("This is a bottle") but fails to explain the chemical reasoning ("Neutralize due to pH < 2").



---

## 💡 Key Features

* 
**🧠 Intelligent Decision Engine:** Uses a **Random Forest Classifier** to map complex chemical properties (pH, flammability, toxicity) to 5 distinct treatment classes.


* **🔍 Explainable AI (XAI):** Integrated with **SHAP (SHapley Additive Explanations)** to provide local and global interpretability. It tells you *exactly* which feature (e.g., High Flammability) triggered the decision.


* 
**🧪 Synthetic Data Generator:** Includes a module to generate realistic hazardous waste datasets based on domain rules, overcoming the scarcity of real-world HHW data.


* 
**♻️ Circular Economy Focus:** Prioritizes **Resource Recovery** and **Neutralization** over landfilling, aligning with the Zero Waste Hierarchy.



---

## 🏗️ System Architecture

The system follows a modular **7-Layer Architecture**:

```mermaid
graph TD
    A[User Interaction Layer] -->|Scenario Inputs| B(Data Simulation Layer)
    B -->|Synthetic Data| C{Feature Engineering}
    C -->|Encoded Vectors| D[Machine Learning Layer]
    D -->|Random Forest| E[Prediction: Treatment Class]
    E --> F[Explainability Layer (SHAP)]
    F -->|Local & Global Plots| G[Visualization Layer]

```

### The Data Flow

The model analyzes the following features:

* `Waste_Type` (Categorical)
* `pH_Level` (Continuous)
* `Flammability_Index` (Continuous)
* `Toxicity_Score` (Continuous)
* `Container_Integrity` (Binary: Intact/Leaking)

---

## 🚀 How It Works (The Logic)

This system moves beyond simple classification. It simulates a logic flow similar to a chemical safety officer:

1. **Input:** User enters data for a "Leaking Battery."
2. **Processing:** The Random Forest model analyzes the `Container_Integrity` and `Chemical_Composition`.
3. **Output:** "Recommendation: **Secure Landfill**."
4. **Explanation (XAI):** "This decision was made because **Container_Integrity = False** and **Toxicity > 8.0**.".



---

## 💻 Installation & Usage

This is a command-line interface (CLI) application.

1. **Clone the repository:**
```bash
git clone https://github.com/AbhikRaj01/HHW_Research.git
cd HHW_Research

```


2. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the System:**
```bash
python main.py

```


4. **View Explanations:**
The system will generate SHAP summary plots (beeswarm) to visualize feature importance across the dataset.



---

## 📊 Research Context

This project addresses specific gaps in current Environmental Engineering literature:

* **The Explainability Void:** Bringing SHAP/LIME frameworks to waste management.


* **The HHW Niche:** shifting focus from industrial waste to the complex, heterogeneous nature of household waste.



*Created as part of an Academic Research Initiative .
