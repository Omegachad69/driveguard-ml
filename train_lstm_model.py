import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# File paths
processed_data_path = os.path.join("data", "processed", "preprocessed_data.csv")
model_path = os.path.join("models", "improved_lstm_cnn_model.h5")

# Load preprocessed data
data = pd.read_csv(processed_data_path)


# Prepare sequences
def create_sequences(data, target_column, time_steps):
    sequences = []
    labels = []
    data_array = data.drop(columns=[target_column]).values
    target_array = data[target_column].values

    for i in range(len(data) - time_steps + 1):
        sequences.append(data_array[i : i + time_steps])
        labels.append(target_array[i + time_steps - 1])

    return np.array(sequences), np.array(labels)


# Hyperparameters
time_steps = 10  # Increased time steps for better temporal context
X, y = create_sequences(data, target_column="Class", time_steps=time_steps)

# Balance the dataset using SMOTE
X_reshaped = X.reshape(X.shape[0], -1)  # Flatten for SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)
X_resampled = X_resampled.reshape(-1, time_steps, X.shape[2])  # Reshape back

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)


# Define the improved LSTM + CNN model
def build_improved_model(input_shape, num_classes):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv1D(
                filters=64, kernel_size=3, activation="relu", input_shape=input_shape
            ),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True)
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# Model setup
input_shape = (X_train.shape[1], X_train.shape[2])
num_classes = len(np.unique(y))
model = build_improved_model(input_shape, num_classes)

# Training
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,  # Increased epochs
    batch_size=64,  # Experiment with batch size
    callbacks=[early_stopping],
)

# Save the model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)

# Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Confusion Matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))

# Training plots
plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Loss Over Epochs")
plt.show()

plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Accuracy Over Epochs")
plt.show()


# import os
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     confusion_matrix,
#     classification_report,
#     ConfusionMatrixDisplay,
# )
# import matplotlib.pyplot as plt
# from imblearn.over_sampling import SMOTE

# # File paths
# processed_data_path = os.path.join("data", "processed", "preprocessed_data.csv")
# model_path = os.path.join("models", "improved_lstm_cnn_model.h5")

# # Load preprocessed data
# data = pd.read_csv(processed_data_path)


# # Prepare sequences
# def create_sequences(data, target_column, time_steps):
#     sequences = []
#     labels = []
#     data_array = data.drop(columns=[target_column]).values
#     target_array = data[target_column].values

#     for i in range(len(data) - time_steps + 1):
#         sequences.append(data_array[i : i + time_steps])
#         labels.append(target_array[i + time_steps - 1])

#     return np.array(sequences), np.array(labels)


# # Hyperparameters
# time_steps = 10  # Increased time steps for better temporal context
# X, y = create_sequences(data, target_column="Class", time_steps=time_steps)

# # Balance the dataset using SMOTE
# X_reshaped = X.reshape(X.shape[0], -1)  # Flatten for SMOTE
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)
# X_resampled = X_resampled.reshape(-1, time_steps, X.shape[2])  # Reshape back

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X_resampled, y_resampled, test_size=0.2, random_state=42
# )


# # Define the improved LSTM + CNN model with regularization
# def build_improved_model(input_shape, num_classes):
#     model = tf.keras.Sequential(
#         [
#             tf.keras.layers.Conv1D(
#                 filters=64,
#                 kernel_size=3,
#                 activation="relu",
#                 input_shape=input_shape,
#                 kernel_regularizer=tf.keras.regularizers.l2(0.01),
#             ),
#             tf.keras.layers.MaxPooling1D(pool_size=2),
#             tf.keras.layers.Bidirectional(
#                 tf.keras.layers.LSTM(
#                     64,
#                     return_sequences=True,
#                     kernel_regularizer=tf.keras.regularizers.l2(0.01),
#                 )
#             ),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.Dropout(0.5),  # Increased dropout
#             tf.keras.layers.Bidirectional(
#                 tf.keras.layers.LSTM(
#                     64, kernel_regularizer=tf.keras.regularizers.l2(0.01)
#                 )
#             ),
#             tf.keras.layers.BatchNormalization(),
#             tf.keras.layers.Dropout(0.5),  # Increased dropout
#             tf.keras.layers.Dense(
#                 32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)
#             ),
#             tf.keras.layers.Dense(num_classes, activation="softmax"),
#         ]
#     )
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Learning rate
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"],
#     )
#     return model


# # Model setup
# input_shape = (X_train.shape[1], X_train.shape[2])
# num_classes = len(np.unique(y))
# model = build_improved_model(input_shape, num_classes)


# # Learning rate scheduler
# def lr_scheduler(epoch, lr):
#     if epoch < 10:
#         return lr
#     else:
#         return lr * tf.math.exp(-0.1)


# lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

# # Early stopping
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor="val_loss", patience=15, restore_best_weights=True
# )

# # Training
# history = model.fit(
#     X_train,
#     y_train,
#     validation_split=0.2,
#     epochs=100,  # Increased epochs
#     batch_size=64,  # Experiment with batch size
#     callbacks=[early_stopping, lr_callback],
#     verbose=1,
# )

# # Save the model
# os.makedirs(os.path.dirname(model_path), exist_ok=True)
# model.save(model_path)

# # Evaluation
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# # Confusion Matrix
# y_pred = np.argmax(model.predict(X_test), axis=1)
# cm = confusion_matrix(y_test, y_pred)
# ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
# plt.title("Confusion Matrix")
# plt.show()

# # Classification Report
# print(classification_report(y_test, y_pred))

# # Training plots
# plt.figure()
# plt.plot(history.history["loss"], label="Training Loss")
# plt.plot(history.history["val_loss"], label="Validation Loss")
# plt.legend()
# plt.title("Loss Over Epochs")
# plt.show()

# plt.figure()
# plt.plot(history.history["accuracy"], label="Training Accuracy")
# plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
# plt.legend()
# plt.title("Accuracy Over Epochs")
# plt.show()
