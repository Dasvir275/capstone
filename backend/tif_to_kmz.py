import gdal
from osgeo import osr
import simplekml

def tif_to_kml(tif_file, output_kml):
    # Open the TIF file
    dataset = gdal.Open(tif_file)

    if dataset is None:
        print("Error: Could not open TIF file.")
        return

    # Get geospatial information
    geotransform = dataset.GetGeoTransform()
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(dataset.GetProjection())

    # Extract metadata
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    ulx, xres, xskew, uly, yskew, yres = geotransform
    lrx = ulx + (width * xres)
    lry = uly + (height * yres)

    # Create KML object
    kml = simplekml.Kml()

    # Create KML polygon
    pol = kml.newpolygon(name="TIF Area")
    pol.outerboundaryis = [(ulx, uly), (lrx, uly), (lrx, lry), (ulx, lry), (ulx, uly)]

    # Set KML style
    pol.style.polystyle.color = simplekml.Color.changealphaint(200, simplekml.Color.green)

    # Save KML file
    kml.save(output_kml)
    print(f"KML file '{output_kml}' created successfully.")

if __name__ == "__main__":
    # Replace 'input.tif' with the path to your TIF file
    tif_file_path = "input.tif"

    # Replace 'output.kml' with the desired KML file path
    kml_output_path = "output.kml"

    tif_to_kml(tif_file_path, kml_output_path)
