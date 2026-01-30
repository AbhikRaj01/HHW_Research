X-IDSS: Explainable AI for Hazardous Waste Management
An Intelligent Decision Support System (IDSS) for Sustainable Household Hazardous Waste (HHW) Treatment.

Abstract
Household Hazardous Waste (HHW)—encompassing items such as lithium batteries, solvents, and corrosive cleaners—presents significant disposal challenges due to the gap between public knowledge and chemical complexity. This project introduces X-IDSS, a system designed to bridge this "Decision Gap." Unlike traditional "Black Box" AI models that merely classify data, this system utilizes Explainable AI (XAI) to recommend safe treatment methods (e.g., Neutralization, Resource Recovery) while providing transparent, chemical-based reasoning for every decision.

Problem Statement
Current waste management frameworks face three critical challenges:

Classification Complexity: End-users often lack the expertise to distinguish between benign items (e.g., zinc-carbon batteries) and volatile hazards (e.g., lithium-polymer batteries).

Environmental Consequence: Improper disposal results in "thermal runaway" events in transport vehicles and long-term groundwater contamination.

The "Black Box" Limitation: Existing AI solutions identify objects but fail to articulate the chemical reasoning required for safety protocols (e.g., necessitating neutralization due to pH < 2).

Core Capabilities
Intelligent Decision Engine
The system employs a Random Forest Classifier to map complex physicochemical properties—including pH, flammability, and toxicity—to five distinct treatment methodologies.

Explainable AI (XAI) Integration
To ensure trust and transparency, the model integrates SHAP (SHapley Additive Explanations). This framework provides both local and global interpretability, identifying precisely which features (e.g., high flammability index) drove specific classification decisions.

Synthetic Data Generation
To address the scarcity of public HHW datasets, the system includes a module that generates realistic hazardous waste data based on established domain rules and chemical constraints.

Circular Economy Alignment
The decision logic prioritizes Resource Recovery and Neutralization over landfill disposal, strictly adhering to the Zero Waste Hierarchy.

System Architecture
The application follows a modular 7-layer architecture designed for scalability and interpretability.

Code snippet
graph TD
    A[User Interaction Layer] -->|Scenario Inputs| B(Data Simulation Layer)
    B -->|Synthetic Data| C{Feature Engineering}
    C -->|Encoded Vectors| D[Machine Learning Layer]
    D -->|Random Forest| E[Prediction: Treatment Class]
    E --> F[Explainability Layer (SHAP)]
    F -->|Local & Global Plots| G[Visualization Layer]
Data Model
The machine learning model analyzes the following feature vector:

Waste_Type (Categorical)

pH_Level (Continuous)

Flammability_Index (Continuous)

Toxicity_Score (Continuous)

Container_Integrity (Binary: Intact vs. Leaking)

Operational Logic
The system simulates the decision-making process of a chemical safety officer:

Input: The system receives specific waste parameters (e.g., a leaking battery scenario).

Processing: The Random Forest model evaluates Container_Integrity alongside Chemical_Composition.

Classification: The system outputs a recommendation, such as Secure Landfill.

Explanation: The XAI layer articulates the rationale: "Decision triggered by Container_Integrity = False AND Toxicity > 8.0."

Installation and Usage
This project is designed as a command-line interface (CLI) application.

Prerequisites
Python 3.8+

pip

Setup
Clone the repository:

Bash
git clone https://github.com/AbhikRaj01/HHW_Research.git
cd HHW_Research
Install Dependencies:

Bash
pip install -r requirements.txt
Execute the System:

Bash
python main.py
Visualization:
Upon execution, the system generates SHAP summary plots (beeswarm) to visualize feature importance and model behavior across the dataset.

Research Significance
This project addresses specific gaps in current Environmental Engineering and Computer Science literature:

The Explainability Void: Applies SHAP/LIME frameworks specifically to the waste management domain.

The HHW Niche: Shifts focus from homogeneous industrial waste to the complex, heterogeneous nature of household waste streams.
