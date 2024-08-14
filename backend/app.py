# Flask app (app.py)
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, storage, firestore
from datetime import datetime
import model
import tif_to_png

app = Flask(__name__)
CORS(app)

# Initialize Firebase
cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred, {"storageBucket": "sih1289-project-web.appspot.com"})

# Initialize Firestore
db = firestore.client()

@app.route('/flood_forecast', methods=['POST'])
def upload_image():
    try:
        data = request.get_json()

        # dischargeVal = 500
        # weather_condition = "monsoon"
        # discharge_or_river_stage = "discharge"

        # for i in range(112):
        #     image_tif_path = model.model(weather_condition, discharge_or_river_stage, dischargeVal)
        #     print(image_tif_path)
        #     image_path = tif_to_png.tifToPng(image_tif_path)
        #     #image_path = tif_to_png.tifToJpg(image_tif_path)
        #     print(image_path)
        #     bucket = storage.bucket()
        #     # Convert number to string before creating the blob path
        #     blob = bucket.blob(f"images/output_image_{weather_condition}_{discharge_or_river_stage}_{target_value}.png")
        #     blob.upload_from_filename(image_path)

        #     blob.make_public()

        #     image_url = blob.public_url

        #     print(image_url)

        #     timestamp = datetime.now()

        #     forecast_data = {
        #         'dischargeValue': dischargeVal,
        #         'riverStageValue': None,
        #         'image_url': image_url,
        #         'timestamp': timestamp
        #     }

        #     db.collection('discharge_data').doc(dischargeVal).set(forecast_data)
        #     print(dischargeVal)
        #     dischargeVal+=500




        weather_condition = "monsoon"
        discharge_or_river_stage = data.get('dischargeOrRiverStage')
        target_value = 0

        if discharge_or_river_stage == 'discharge':
            discharge_value = data.get('dischargeValue')
            print(discharge_value)
            river_stage_value = None  # Set river_stage_value to null
            target_value = discharge_value
        elif discharge_or_river_stage == 'stage':
            river_stage_value = data.get('riverStageValue')
            print(river_stage_value)
            discharge_value = None  # Set discharge_value to null
            target_value = river_stage_value



        # Assuming the image is present in the current working directory with a fixed name "example.jpg"
        image_tif_path = model.model(weather_condition, discharge_or_river_stage, target_value)
        print(image_tif_path)
        image_path = tif_to_png.tifToPng(image_tif_path)
        #image_path = tif_to_png.tifToJpg(image_tif_path)
        print(image_path)

        print(f"Image upload request.")

        # Upload image to Firebase Storage
        bucket = storage.bucket()
        # Convert number to string before creating the blob path
        blob = bucket.blob(f"images/output_image_{weather_condition}_{discharge_or_river_stage}_{target_value}.png")
        blob.upload_from_filename(image_path)

        blob.make_public()

        image_url = blob.public_url

        print(image_url)

        timestamp = datetime.now()

        forecast_data = {
            'dischargeValue': discharge_value,
            'riverStageValue': river_stage_value,
            'image_url': image_url,
            'timestamp': timestamp
        }

        db.collection('forecast').add(forecast_data)

        return jsonify({'success': True, 'image_url': image_url})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)