import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import tifffile as tiff
from skimage.transform import resize
from skimage.filters import gaussian
import joblib
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import median_filter
from skimage.filters import threshold_otsu

def model(weather_condition,interpolation_variable, target_value):
    scaler = MinMaxScaler()

    # Choose the CSV file based on the weather condition
    if weather_condition == "monsoon":
        csv_file = r"Sorted_Dataset\SIH_wet_discharge_stage_data.csv"
    elif weather_condition == "dry_weather":
        csv_file = r"Sorted_Dataset\SIH_Dry_discharge_stage_data.csv"
    else:
        print("Invalid weather condition. Exiting.")
        exit()

    data = pd.read_csv(csv_file)  # Adjust the delimiter based on your CSV format

    # Define target size for resizing

    target_size = (224, 224)


    if weather_condition == "monsoon":
        base_image_directory = r"Sorted_Dataset\Monsoon"
        output_tiff_directory = r"Sorted_Dataset\monsoon_cleaned_images"
    elif weather_condition == "dry_weather":
        base_image_directory = r"Sorted_Dataset\Dry Weather"
        output_tiff_directory = r"Sorted_Dataset\dry_weather_cleaned_images"
    else:
        print("Invalid weather condition. Exiting.")
        exit()

    if weather_condition == "monsoon":
        model_path = r"Sorted_Dataset\monsoon_sar_model.h5"
        scaler_path = r"Sorted_Dataset\monsoon_discharge_scaler.pkl"
    else:
        model_path = r"Sorted_Dataset\dry_weather_sar_model.h5"
        scaler_path = r"Sorted_Dataset\dry_weather_discharge_scaler.pkl"

    
    loaded_model = keras.models.load_model(model_path)
    loaded_scaler = joblib.load(scaler_path)

    data.columns = data.columns.str.strip()
    
    # Normalize the interpolation values
    if interpolation_variable == "discharge":
        y = scaler.fit_transform(data['Discharge'].values.reshape(-1, 1))
    elif interpolation_variable == "stage":
        y = scaler.fit_transform(data['Stage'].values.reshape(-1, 1))
    else:
        print("Invalid interpolation variable. Exiting.")
        exit()
    
    # Inverse transform to obtain actual interpolation value
    normalized_target_value = scaler.transform(np.array(target_value).reshape(1, -1))[0, 0]
    closest_index = np.argmin(np.abs(y - normalized_target_value))
    closest_interpolation_value = scaler.inverse_transform(y[closest_index:closest_index+1])[0, 0]
    closest_image_name = data.loc[closest_index, 'Image Name']
    # Remove the serial number from the image name
    closest_image_name = closest_image_name.split(':')[-2].strip()

    closest_image_path = os.path.join(base_image_directory, f"{closest_image_name}.tif")
    
    

    # Reshape interpolated image for display  
    # Now, closest_image_path contains the full path to the image
    #print(f"Path to closest image for Target {interpolation_variable.capitalize()}: {closest_image_path}")

    # Preprocess the image
    target_image = preprocess_image(closest_image_path, target_size)

    # Check if the image is None
    if target_image is None:
        print("Image could not be processed. Exiting.")
    else:
        # Reshape the input image for model prediction
        target_image = np.expand_dims(target_image, axis=0)

        # Make the prediction
        predicted_interpolation_value = loaded_model.predict(target_image)


        # Visualize the input and cleaned images
        # plt.figure(figsize=(10, 4))

        # plt.subplot(1, 2, 1)
        # plt.imshow(target_image[0], cmap='gray')  # Assuming it's a grayscale image
        # plt.title(f'Interpolated Image : {interpolation_variable.capitalize()} : {target_value} : {weather_condition}')

        # plt.subplot(1, 2, 2)
        cleaned_image = clean_image(target_image[0], target_image[0]) 
        # plt.imshow(cleaned_image, cmap='gray')  # Assuming it's a grayscale image
        # plt.title(f'Cleaned Image ')

        # plt.show()

        # Print information about the closest image in the dataset
        print(f"Closest Image Name: {closest_image_name}")
        print(f"Closest {interpolation_variable.capitalize()} Value: {closest_interpolation_value}")

        original_tif_path = closest_image_path
        #save_tiff(cleaned_image, weather_condition,interpolation_variable, target_value, original_tif_path)
        save_tiff(cleaned_image, weather_condition,interpolation_variable, target_value, original_tif_path)
        return rf"Output_Images\output_image_{weather_condition}_{interpolation_variable}_{target_value}.tif"


def save_tiff(image, weather_condition,interpolation_variable, target_value, original_tif_path):
    try:
        #os.makedirs(output_directory, exist_ok=True)
        os.makedirs("Output_Images", exist_ok=True)
        if target_value is not None:
            target_value = int(target_value)
        else:
            target_value = 0
        #output_filename = f"cleaned_image_discharge_{discharge_value}.tif"
        output_filename = f"output_image_{weather_condition}_{interpolation_variable}_{target_value}.tif"
        output_path = os.path.join("Output_Images", output_filename)


        # Load spatial reference information from the original TIF using rasterio
        with rasterio.open(original_tif_path) as src:
            profile = src.profile
            transform = src.transform

            # Update the profile for the cleaned image
            profile.update(
                dtype=np.float32,
                count=3,
                photometric='rgb',
                compress='deflate',  # You can adjust compression as needed
            )

            # Write the image using rasterio to ensure compatibility and transfer spatial information
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(image.transpose(2, 0, 1).astype(np.float32))  # Transpose to (bands, rows, columns)

        print(f"Cleaned image saved successfully at: {output_path}")
    except Exception as e:
        print(f"Error saving the cleaned image: {e}")


def post_process_image(cleaned_image):
    # Apply median filtering to smooth the image
    smoothed_image = np.zeros_like(cleaned_image)
    for c in range(3):
        smoothed_image[:, :, c] = median_filter(cleaned_image[:, :, c], size=3)  # Adjust the size as needed

    # Use Otsu's thresholding to segment the image into foreground and background
    threshold_value = threshold_otsu(smoothed_image)
    binary_mask = smoothed_image > threshold_value

    # Apply the binary mask to the cleaned image to remove small scattered pixels
    cleaned_image_post_processed = cleaned_image * binary_mask

    return cleaned_image_post_processed


def preprocess_image(image_path, target_size):
    try:
        img = tiff.imread(image_path)
        img_resized = resize(img, target_size, anti_aliasing=True)
        return img_resized
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def clean_image(input_image, ideal_image, density_threshold=0.2, post_process=True):
    cleaned_image = np.copy(input_image)

    # Check if ideal_image is a grayscale image
    if len(ideal_image.shape) == 2:
        # Calculate the threshold for pixel density based on the ideal image
        pixel_density_threshold = np.max(ideal_image) * density_threshold

        # Identify the main river stream by thresholding
        main_river_stream = input_image >= pixel_density_threshold

        # Set pixels outside the main river stream to 0 (black)
        cleaned_image[~main_river_stream] = 0
    else:
        # For RGB images, iterate over each channel
        for c in range(input_image.shape[2]):
            # Calculate the threshold for pixel density based on the ideal image
            pixel_density_threshold = np.max(ideal_image[:, :, c]) * density_threshold

            # Identify the main river stream by thresholding
            main_river_stream = input_image[:, :, c] >= pixel_density_threshold

            # Set pixels outside the main river stream to 0 (black)
            cleaned_image[~main_river_stream, c] = 0

        # Apply post-processing if specified
    if post_process:
        cleaned_image = post_process_image(cleaned_image)

    return cleaned_image


def interpolate_image(target_value, interpolation_values, images):
    # Check if the target value is within the range of available values
    target_value = float(target_value)
    min_value, max_value = np.min(interpolation_values), np.max(interpolation_values)
    
    print(f"Target value: {target_value}")
    print(f"Min available value: {min_value}, Max available value: {max_value}")

    if target_value < min_value or target_value > max_value:
        print(f"Target value {target_value} is outside the range of available values.")
        return None

    # Reshape images to 2D array for interpolation
    flattened_images = images.reshape((images.shape[0], -1))

    # Perform linear interpolation for each channel separately
    interpolated_channels = []
    for channel in range(flattened_images.shape[-1]):
        interpolated_channel = np.interp(target_value, interpolation_values, flattened_images[:, channel])
        interpolated_channels.append(interpolated_channel)

    # Stack the channels to form the final interpolated image
    interpolated_image = np.stack(interpolated_channels, axis=-1)

    return interpolated_image