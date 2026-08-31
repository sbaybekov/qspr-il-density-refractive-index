---
title: QSPR IL Density and Refractive Index
emoji: 🧪
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.38.0
app_file: app.py
pinned: false
---

# QSPR: Ionic-Liquid Density & Refractive Index

Predict density (kg/m3) and refractive index (Na D-line) of ionic-liquid
mixtures (in water, ethanol, or isopropanol) and pure ionic liquids, using
trained XGBoost ensemble models.

Choose a property and system in the sidebar, then either upload a CSV of
ionic-liquid SMILES or enter a single SMILES string directly.

Source code, model training details, and the underlying data pipeline are
documented in the main project repository:
https://github.com/kammmran/qspr-il-density-refractive-index
