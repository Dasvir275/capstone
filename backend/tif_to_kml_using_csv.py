import rasterio
import csv
import simplekml
import pandas as pd
from utm import to_latlon

def tiff_to_csv(tiff_file, csv_file):
    with rasterio.open(tiff_file) as src:
        data = src.read(1)  # Assuming a single-band TIFF
        height, width = data.shape

        with open(csv_file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            # Write CSV header
            csvwriter.writerow(['latitude', 'longitude', 'value'])

            # Write CSV data
            for i in range(height):
                for j in range(width):
                    latitude, longitude = src.xy(i, j)
                    value = data[i, j]
                    csvwriter.writerow([latitude, longitude, value])

            extract_image_data('input.tif')



def extract_image_data(image_path):
    # Open the GeoTIFF image with rasterio
    with rasterio.open(image_path) as src:
        # Read the image data as a numpy array
        image_data = src.read(1)

        # Get image size
        height, width = image_data.shape

        # Initialize lists to store data
        pixel_data = []

        # Get the affine transformation coefficients
        transform = src.transform

        # Iterate through each pixel
        for y in range(height):
            for x in range(width):
                # Get pixel value (0 or 1 for black or white in a typical mask)
                pixel_value = image_data[y, x]

                # Convert pixel coordinates to geographical coordinates (lat, lon)
                lon, lat = transform * (x + 0.5, y + 0.5)
                lat,lon=to_latlon(lon, lat, 44, 'N')
                pixel_data.append([y * width + x + 1, lat, lon, pixel_value])

    # Convert the list to a pandas DataFrame
    df = pd.DataFrame(pixel_data, columns=['Pixel_Number', 'Lat', 'Lon', 'Pixel_Value'])
    df.to_csv('coordinates.csv', index=False)
    csv_to_kml("coordinates.csv", 'output.kml')

# Example usage:



def csv_to_kml(csv_file, kml_file):
    kml = simplekml.Kml()

    with open(csv_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # Skip the header

        for row in csvreader:
            latitude, longitude, value = map(float, row)
            kml.newpoint(name=str(value), coords=[(longitude, latitude)])

    kml.save(kml_file)


# tiff_file = 'input.tif'
# csv_file = 'output.csv'
# tiff_to_csv(tiff_file, csv_file)

csv_file = 'coordinates.csv'
kml_file = 'output.kml'
csv_to_kml(csv_file, kml_file)

