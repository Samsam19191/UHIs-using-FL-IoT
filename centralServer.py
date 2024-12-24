from threading import Thread, Lock
from queue import Queue
from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from model import WeatherModel
import csv


# Function to aggregate model weights using pth files
def aggregate_weights(model_paths):
    """Aggregate the weights of multiple models (from different sensors) using pth files."""
    # Load the weights from the first model
    aggregated_weights = torch.load(model_paths[0])

    # Iterate over the model paths and average their weights
    for key in aggregated_weights:
        # Sum up all weights for each parameter
        weight_sum = torch.zeros_like(aggregated_weights[key])
        for path in model_paths:
            model_weights = torch.load(path)
            weight_sum += model_weights[key]

        # Average the weights
        aggregated_weights[key] = weight_sum / len(model_paths)

    return aggregated_weights


def makePrediction(data, model):
    """Make a prediction using the model"""
    # Remove the last item from each row of data
    data = [row[:-1] for row in data]
    # Convert the data to a tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # Ensure the model is in evaluation mode
    model.eval()

    # Make predictions
    with torch.no_grad():
        predictions = model(data_tensor)

    return predictions


def vizualisation(data, predictions):
    """Visualize the data and predictions"""
    # Extract the 3rd item from each data row
    actual_values = [float(row[3]) for row in data]

    # Convert predictions to a list
    predicted_values = predictions.numpy().tolist()

    # Plot the actual vs predicted values
    plt.figure(figsize=(10, 5))
    plt.plot(actual_values, label="Actual Values")
    plt.plot(predicted_values, label="Predicted Values", linestyle="--")
    plt.xlabel("Data Point Index")
    plt.ylabel("Value")
    plt.title("Actual vs Predicted Values")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    model = WeatherModel()
    model_paths = [f"models/{i}_weather_model.pth" for i in range(8)]
    model.load_state_dict(aggregate_weights(model_paths))
    with open("data/eval_data.csv", newline="") as csvfile:
        datareader = csv.reader(csvfile)
        data = [row for row in datareader]
    data = data[1:]
    data = [[float(item) for item in row] for row in data]
    predictions = makePrediction(data, model)
    vizualisation(data, predictions)
