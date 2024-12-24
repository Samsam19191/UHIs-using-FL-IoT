# Federated Learning in IoT: Heat Island Detection

This project implements a federated learning approach to predict urban heat islands using simulated weather data. Individual IoT devices (stations) train local models, and a central server aggregates their weights into a global model while maintaining data privacy.

## Features
- Local training on simulated weather data.
- Federated weight aggregation.
- Prediction of heat islands (output: 0 or 1).
- Visualization of predictions and accuracy.

## Tech Stack
- Python
- PyTorch
- Matplotlib

## How It Works
1. Each station trains a model on local data.
2. A global model aggregates weights from all stations.
3. The global model predicts heat islands based on new data.
