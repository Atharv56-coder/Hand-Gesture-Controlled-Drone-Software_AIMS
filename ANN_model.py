import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.read_csv('new_gesture_dataset_500_2.csv')

df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(str)


X = df.drop('label', axis=1).values
y = df['label'].values

unique_labels = np.unique(y)
print(f"Training on classes: {unique_labels}")

label_map = {label: i for i, label in enumerate(unique_labels)}
y_encoded = np.array([label_map[l] for l in y])
y_one_hot = tf.keras.utils.to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

model = models.Sequential([
    layers.Input(shape=(X.shape[1],)), 
    
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2), 
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    
    layers.Dense(32, activation='relu'),
    
    layers.Dense(len(unique_labels), activation='softmax') 
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_data=(X_test, y_test),
    verbose=1
)

model.save('hand_gesture_model_2_test.h5')
print("Model saved as hand_gesture_model.h5")