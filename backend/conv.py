from PIL import Image
import numpy as np
import simplekml

def tif_to_kml(tif_path, kml_path):
    # Open TIFF image using PIL
    image = Image.open(tif_path)

    # Convert image to NumPy array
    image_array = np.array(image)

    # Get image dimensions
    height, width = image_array.shape[:2]

    # Create KML object
    kml = simplekml.Kml()

    # Loop through each pixel in the image
    for y in range(height):
        for x in range(width):
            # Get the pixel value
            pixel_value = image_array[y, x]

            # Create a KML point for each pixel
            kml.newpoint(name=f'Pixel ({x}, {y})', coords=[(x, y)])

            # You may want to customize the KML point based on the pixel value
            # For example, you can set the point color or other properties

    # Save KML file
    kml.save(kml_path)

# Example usage
tif_file_path = r'Output_Images\output_image_monsoon_999.tif'
kml_file_path = 'output.kml'

tif_to_kml(tif_file_path, kml_file_path)

  

# from PIL import Image
# import simplekml

# def convert_image_to_kml(image_path, output_kml_path):
#     kml = simplekml.Kml()

#     # Open the image
#     with Image.open(image_path) as img:
#         width, height = img.size

#         # Iterate through each pixel in the image
#         for x in range(width):
#             for y in range(height):
#                 # Get the pixel color (assuming RGB or grayscale image)
#                 pixel_color = img.getpixel((x, y))

#                 # Check if the pixel is white (255 for grayscale, (255, 255, 255) for RGB)
#                 if isinstance(pixel_color, int) and pixel_color == 255 or \
#                    isinstance(pixel_color, tuple) and all(value == 255 for value in pixel_color):
#                     # Add a KML point for the white pixel
#                     point = kml.newpoint()
#                     point.coords = [(x, y)]

#     # Save the KML file
#     kml.save(output_kml_path)

# # Specify the path to your image and the output KML file
# image_path = "test2.png"
# output_kml_path = "output.kml"

# # Convert image to KML
# convert_image_to_kml(image_path, output_kml_path)

