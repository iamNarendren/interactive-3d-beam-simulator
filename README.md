# 🏗️ Structural Intelligence Lab
### From Physics to Prediction — Interactive 3D Structural Deformation

An interactive engineering simulation combining the **Finite Element Method (FEM)**, **Proper Orthogonal Decomposition (POD)**, and machine learning to explore faster structural deformation prediction.

🌐 **[Launch the Live Simulator](https://interactive-3d-beam-simulator.streamlit.app/)**

---

## 🚀 Project Overview

Structural Intelligence Lab is an interactive simulation platform developed for the study of structural deformation in Computational Mechanics.

It allows users to modify beam properties, apply different forces, visualize deformation in 3D, and compare a full FEM reference solution with a reduced-order machine learning (ROM-ML) prediction.

### 🎯 Project Objectives

- Visualize cantilever beam deformation interactively.
- Explore how material properties, beam dimensions, and applied force affect displacement.
- Reduce the dimensionality of FEM simulation data using POD/SVD.
- Use machine learning to predict reduced-order coefficients.
- Compare ROM-ML predictions against FEM reference results.

---

## ✨ Key Features

- **Interactive 3D Beam:** Rotate and explore the beam visualization.
- **Material Selection:** Choose between Steel, Iron, and Tungsten.
- **Adjustable Parameters:** Modify force, beam length, width, thickness, and visual deformation magnification.
- **Animated Deformation:** Observe the computed beam deformation.
- **FEM vs. ROM-ML:** Compare the reference solution with the machine learning prediction.
- **Error Analysis:** Inspect displacement plots and prediction error metrics.
- **Runtime Comparison:** View the reported computational time for the compared methods.
- **Dark Engineering Interface:** A black background with a blue-to-yellow-to-red beam colour gradient.

---

## 🔬 Methodology

The application follows this computational workflow:

**FEM Simulation → POD/SVD → Machine Learning → Reconstruction → Comparison**

### 1. Finite Element Method (FEM)
The beam-element FEM model calculates the reference displacement of a cantilever beam under the selected loading and material conditions.

### 2. Proper Orthogonal Decomposition (POD)
Singular Value Decomposition (SVD) is used to identify a reduced representation of the displacement data.

### 3. Machine Learning
A regression model predicts the reduced-order coefficients from the input beam parameters.

### 4. Displacement Reconstruction
The predicted coefficients are used to reconstruct an approximate displacement field.

### 5. Performance Evaluation
The reconstructed prediction is compared with the FEM reference using displacement errors and reported runtime measurements.

---

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| Streamlit | Interactive web interface |
| Plotly | 3D visualization, plots, and animation |
| NumPy | Numerical arrays and calculations |
| SciPy | Numerical computation and linear algebra |
| Scikit-learn | Machine learning regression |

---

## 🌐 Live Demonstration

**Website:** [https://interactive-3d-beam-simulator.streamlit.app/](https://interactive-3d-beam-simulator.streamlit.app/)

Open the live application to:

1. Select a beam material.
2. Adjust the applied force and beam dimensions.
3. Explore the 3D deformation visualization.
4. Compare the FEM reference and ROM-ML prediction.
5. Inspect the displacement and error information.

---

## 📐 Model Scope and Limitations

- The beam is visualized in 3D using a beam-element FEM solution.
- The application is **not a full 3D solid-element FEM solver**.
- The ROM-ML prediction is an approximation of the FEM reference.
- Prediction accuracy depends on the training data and selected input parameters.
- A prediction within the training range does not guarantee a small error.
- Runtime comparisons depend on the implementation and execution environment.

These limitations are important when interpreting simulation results.

---

## 💻 Run Locally

### Prerequisites
- Python installed on your computer
- Git installed (if cloning the repository)

### 1. Clone the repository

Replace `YOUR_GITHUB_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/interactive-3d-beam-simulator.git
cd interactive-3d-beam-simulator
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment on Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Launch the application

```bash
python -m streamlit run app.py
```

Open the local address displayed in the terminal, usually `http://localhost:8501`.


---

## 🎓 Academic Information

**Project Title:** Physics-Guided Reduced-Order Machine Learning for Fast Structural Deformation Prediction

**Subject:** Computational Mechanics

**Project Type:** Interactive numerical simulation and reduced-order machine learning demonstration

---

*Structural Intelligence Lab — Connecting physics-based simulation with data-driven prediction.*

