from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import tifffile 
import imageio
from skimage import io

def tifToJpg(tif_path):
    img = io.imread(tif_path)
    print(1)
    io.imsave("image.png", img)
    print(2)
    return r"image.png"

def tifToPng(tif_path):
    # Load the TIFF image
    tif_path = rf"{tif_path}"
    #tif_path = r"Sorted_Dataset\Monsoon\2016-07-04.tif"
    image = tifffile.imread(tif_path)

    # Convert the TIFF image to PNG
    png_path = r"image.png"
    imageio.imwrite(png_path, image.astype(np.uint8))  # Convert to uint8

    # Convert to uint8 before creating the Pillow image
    tif_image = Image.fromarray(image.astype(np.uint8))

    # Convert black color to transparent and white color to #00A9FF
    image_array = np.array(tif_image.convert("RGBA"))

    # Define the black and white colors
    black_color = np.array([0, 0, 0, 255])  # Black in RGBA
    white_color = np.array([255, 255, 255, 255])  # White in RGBA

    # Create a mask for black pixels
    black_mask = np.all(image_array[:, :, :3] == black_color[:3], axis=-1)

    # Apply the mask to set black pixels to transparent
    image_array[black_mask] = [0, 0, 0, 0]

    # Create a mask for white pixels
    white_mask = np.all(image_array[:, :, :3] == white_color[:3], axis=-1)

    # Apply the mask to set white pixels to #00A9FF
    image_array[white_mask, :3] = [255, 0, 0]

    # Save the final image
    output_image = Image.fromarray(image_array)
    output_image.save(png_path, format="PNG",quality=95)

    # Load the PNG image with transparency
    png_path = "image.png"
    image_with_rounded_edges = Image.open(png_path)

    # Add rounded edges with a specified radius (adjust as needed)
    radius = 90
    image_with_rounded_edges = add_rounded_edges(image_with_rounded_edges, radius)

    # Save the final image with rounded edges
    output_path_with_rounded_edges = "image_with_rounded_edges.png"
    image_with_rounded_edges.save(output_path_with_rounded_edges, format="PNG", quality=95)

    # Apply sharpening using the unsharp mask filter
    sharpened_image_array = Image.fromarray(image_array).filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    # Save the final sharpened image
    sharpened_image_array.save(png_path, format="PNG")

    return r"image.png"

def add_rounded_edges(image, radius):
    # Create a mask for the rounded edges
    width, height = image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), (width, height)], radius, fill=255)
    mask_array = np.array(mask)

    # Apply the mask to the image
    image_array = np.array(image)
    image_array[mask_array == 0] = [0, 0, 0, 0]

    return Image.fromarray(image_array, "RGBA")


