# from flask import Flask, request, jsonify
# import numpy as np
# import tensorflow as tf
# from datetime import datetime

# app = Flask(__name__)

# # Load the trained model
# model = tf.keras.models.load_model("improved_lstm_cnn_model.h5")

# # Global variables to store real-time data and predictions
# real_time_data = []
# predictions = []


# # Endpoint to receive sensor data
# @app.route("/send_data", methods=["POST"])
# def send_data():
#     global real_time_data, predictions

#     try:
#         # Get the data from the POST request
#         data = request.json["data"]
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#         # Ensure the data has the correct shape (10 time steps, 14 features)
#         if len(data) != 10 or any(len(row) != 14 for row in data):
#             return jsonify({"error": "Invalid data shape. Expected (10, 14)."}), 400

#         # Append the data with timestamp
#         real_time_data.append({"timestamp": timestamp, "data": data})

#         # Convert data to numpy array for prediction
#         data_array = np.array(data).reshape(1, 10, -1)  # Reshape to (1, 10, 14)

#         # Make a prediction
#         prediction = model.predict(data_array)
#         predicted_class = np.argmax(prediction, axis=1)[0]

#         # Store the prediction
#         predictions.append({"timestamp": timestamp, "prediction": int(predicted_class)})

#         # Return the prediction as a JSON response
#         return jsonify({"status": "success", "predicted_class": int(predicted_class)})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# # Endpoint to calculate the driver's score after the journey
# @app.route("/calculate_score", methods=["GET"])
# def calculate_score():
#     global predictions

#     if not predictions:
#         return jsonify({"error": "No data available for scoring"}), 400

#     try:
#         # Calculate the driver's score (example: percentage of safe driving predictions)
#         safe_driving_count = sum(
#             1 for p in predictions if p["prediction"] == 0
#         )  # Assuming 0 is the safe driving class
#         total_predictions = len(predictions)
#         driver_score = (safe_driving_count / total_predictions) * 100

#         # Clear the data for the next journey
#         real_time_data.clear()
#         predictions.clear()

#         return jsonify({"driver_score": driver_score})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)


from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from datetime import datetime

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("improved_lstm_cnn_model.h5")

# Global variables to store real-time data and predictions
real_time_data = []
predictions = []


# Endpoint to receive sensor data
@app.route("/send_data", methods=["POST"])
def send_data():
    global real_time_data, predictions

    try:
        # Get the data from the POST request
        data = request.json["data"]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Ensure the data has the correct shape (10 time steps, 14 features)
        if len(data) != 10 or any(len(row) != 14 for row in data):
            return jsonify({"error": "Invalid data shape. Expected (10, 14)."}), 400

        # Append the data with timestamp
        real_time_data.append({"timestamp": timestamp, "data": data})

        # Convert data to numpy array for prediction
        data_array = np.array(data).reshape(1, 10, -1)  # Reshape to (1, 10, 14)

        # Make a prediction
        prediction = model.predict(data_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Map the predicted class to driving behavior
        driving_behavior = (
            "Normal Driving" if predicted_class == 1 else "Aggressive Driving"
        )

        # Store the prediction
        predictions.append({"timestamp": timestamp, "prediction": driving_behavior})

        # Return the prediction as a JSON response
        return jsonify({"status": "success", "driving_behavior": driving_behavior})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Endpoint to calculate the driver's score after the journey
@app.route("/calculate_score", methods=["GET"])
def calculate_score():
    global predictions

    if not predictions:
        print("No predictions available for scoring.")
        return jsonify({"error": "No data available for scoring"}), 400

    try:
        normal_driving_count = sum(
            1 for p in predictions if p["prediction"] == "Normal Driving"
        )
        total_predictions = len(predictions)
        driver_score = (normal_driving_count / total_predictions) * 100

        print(f"Predictions: {predictions}")
        print(f"Normal Driving Count: {normal_driving_count}")
        print(f"Total Predictions: {total_predictions}")
        print(f"Driver Score: {driver_score}")

        # Clear the data for the next journey
        real_time_data.clear()
        predictions.clear()

        return jsonify({"driver_score": driver_score})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
