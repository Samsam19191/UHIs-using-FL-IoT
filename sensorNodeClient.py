import random
import datetime
import socket
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from model import WeatherModel


class SensorNode:

    def __init__(self, stationId, sendingNumber=10):
        self.sendingNumber = sendingNumber
        self.stationId = stationId
        dataWithoutY = self.generateSensorsData()
        print("All the data is generated by sensor node", self.stationId)

        self.data = self.addYtoData(dataWithoutY=dataWithoutY)

        print("The model is ready to be trained", self.stationId)

        self.model = WeatherModel()
        self.model = self.trainModel()
        # Save the trained model
        torch.save(self.model.state_dict(), f"models/{self.stationId}_weather_model.pth")

    def preprocess_data(self, data):
        # Extract features and labels
        X = [[d[3], d[4], d[5]] for d in data]  # temperature, month, condition
        y = [d[6] for d in data]  # isHeatIsland

        # Encode categorical data
        le = LabelEncoder()
        X = [[x[0], x[1], le.fit_transform([x[2]])[0]] for x in X]

        return X, y

    def trainModel(self):
        X, y = self.preprocess_data(self.data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Training loop
        epochs = 5000
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        # Evaluate the model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            accuracy = (predicted == y_test).sum().item() / y_test.size(0)
            print(f"Accuracy: {accuracy:.4f}")

        return self.model

    def generateSensorsData(self):
        # Generate random data
        dataWithoutY = []

        for i in range(self.sendingNumber):
            longitude = stationDic[self.stationId][0]
            latitude = stationDic[self.stationId][1]
            month = random.randint(1, 12)
            if self.stationId <= (len(stationsCoords) - 1) / 2:
                if month <= 2 and month >= 11:
                    temperature = random.randint(5, 15)
                else:
                    temperature = random.randint(18, 35)
            else:
                if month <= 2 and month >= 11:
                    temperature = random.randint(-3, 13)
                else:
                    temperature = random.randint(15, 30)
            condition = random.randint(0, 3)

            dataWithoutY.append(
                [self.stationId, latitude, longitude, temperature, month, condition]
            )

        return dataWithoutY

    def addYtoData(self, dataWithoutY):
        weather_data = pd.read_csv("data/Istanbul_Weather_Data.csv")
        months = []
        for date in weather_data["DateTime"]:
            month = date.split(".")[1]
            months.append(month)
        weather_data["DateTime"] = months
        weather_data["AvgTemp"] = (
            weather_data["MaxTemp"] + weather_data["MinTemp"]
        ) / 2
        for data in dataWithoutY:
            condition = data[5]
            month = f"{data[4]:02d}"
            relevant_data = weather_data[
                (weather_data["DateTime"] == month)
                & (weather_data["Condition"] == conditions[condition])
            ]
            temperature = data[3]
            avg_temp = relevant_data["AvgTemp"].mean()
            if temperature - avg_temp >= 2:
                data.append(1)
            else:
                data.append(0)

        return dataWithoutY


if __name__ == "__main__":

    stationsCoords = [
        (41.133, 29.067),
        (41.25, 29.033),
        (40.9, 29.15),
        (40.977, 28.821),
        (40.97, 28.82),
        (40.9, 29.31),
        (40.899, 29.309),
        (40.667, 29.283),
    ]
    stationDic = {i: station for i, station in enumerate(stationsCoords)}
    conditions = ["Sunny", "Partly cloudy", "Light rain shower", "Moderate snow"]
    sendingNumber = 1000  # Number of data to send

    stationId = int(input(f"Enter the stationID you are (0-{len(stationsCoords)-1}): "))
    SensorNode(stationId, sendingNumber)
