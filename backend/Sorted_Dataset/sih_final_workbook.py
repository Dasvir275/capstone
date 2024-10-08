import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tifffile as tiff
from skimage.transform import resize
import joblib

# Prompt user to choose the folder
chosen_folder = input("Enter the folder name (e.g., 'Dry Weather' or 'Monsoon'): ")

# Load CSV data
csv_file = r"C:\Users\Lenovo\Downloads\SIH_data_withdischarge.csv"
data = pd.read_csv(csv_file)

# Remove serial numbers from file names
data['Image Name'] = data['Image Name'].str.extract(r'(\d{4}-\d{2}-\d{2})')

# Image directory based on user choice
base_image_directory = os.path.join(r"E:\Sorted_Dataset", chosen_folder)

# Define target size for resizing
target_size = (224, 224)  # Change this as needed

# Define a function to preprocess TIF images
def preprocess_image(image_path):
    try:
        # Read TIF image
        img = tiff.imread(image_path)
        
        # Resize image to the target size
        img_resized = resize(img, target_size, anti_aliasing=True)
        
        # Add any additional preprocessing steps as needed
        
        return img_resized
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Preprocess images and load into a numpy array
X = []
y = []
c =0
for filename in os.listdir(base_image_directory):
    if filename.endswith('.tif'):
        # Construct the full path to the image based on the base directory
        image_path = os.path.join(base_image_directory, filename)

        # Preprocess the TIF image
        img_array = preprocess_image(image_path)
        
        if img_array is not None:
            # Extract the date part from the file name
            image_name = filename.split('.')[0]
            # Find the corresponding record in the CSV file
            matching_record = data[data['Image Name'] == image_name]
            if not matching_record.empty:
                c= c+1
                discharge_value = matching_record['Discharge'].values[0]
                X.append(img_array)
                y.append(discharge_value)
print(f"Total NonEmpty records are {c}")
X = np.array(X)
y = np.array(y)

# Normalize discharge values
scaler = MinMaxScaler()
y_normalized = scaler.fit_transform(y.reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_normalized, test_size=0.2, random_state=42)

# Define the CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], X.shape[3])),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
mse, mae = model.evaluate(X_test, y_test)
print(f'Mean Squared Error: {mse}, Mean Absolute Error: {mae}')

# Save the trained model and scaler for future use
model.save(f'{chosen_folder.lower().replace(" ", "_")}_sar_model.h5')
joblib.dump(scaler, f'{chosen_folder.lower().replace(" ", "_")}_discharge_scaler.pkl')
