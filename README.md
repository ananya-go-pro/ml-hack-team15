# PathSnap: Distributed Map-Matching & Route Reconstruction
**Team 15** | PES University — Machine Learning Hackathon (UE24CS352A)

## Project Overview
PathSnap is an end-to-end machine learning system that reconstructs realistic vehicle trajectories by matching noisy GPS observations to an urban road network.

## Current Progress: Phase 1 (EDA & Data Exploration)
The dataset exploration and statistical verification (P1 – Adarsh) is complete:
- **`eda/eda_pipeline.py`**: Automated dataset exploration, data cleaning verification, anomaly detection, and Folium map generator.
- **`eda/outputs/EDA_REPORT.md`**: Comprehensive statistical report and quality findings across all 90 trips (80 training, 10 student-test).
- **`eda/outputs/dataset_inventory.csv`**: Partition-level inventory and summary counts.
- **`eda/outputs/trips_statistics.csv`**: Granular per-trip statistics (observation count, duration, distance, speed).
- **`eda/outputs/anomalies_detected.csv`**: Register of detected GPS dropouts (>2s) and temporal discontinuities.
- **`eda/outputs/*.png`**: Diagnostic plots (GPS points distribution, sampling interval histogram, road network complexity, spatial overlay).
- **`eda/outputs/interactive_trip_folium.html`**: Interactive Folium map layer showing raw GPS points, road network geometry, and ground truth route.

## Dataset Integrity Summary
- **Missing / Corrupted Values:** 0 across all 90 trips.
- **Topological Integrity:** 100% valid node and arc references.
- **Nominal Sampling Rate:** 94.32% at exactly $\Delta t = 1.0\text{ s}$ (1 Hz).
