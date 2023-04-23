from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="C:\\Users\\Admin\\Downloads\\model.tflite")
interpreter.allocate_tensors()

# Define API endpoint for receiving image data
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load image from request
        image_file = request.files['image']
        image = Image.open(image_file)

        # Preprocess the image
        image = image.resize((224, 224))  # Resize to the input size of the model
        image = np.asarray(image)
        image = (image / 255.0).astype(np.float32)  # Normalize pixel values to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Run inference with the TFLite model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # Process the prediction results
        # You can convert the prediction results to the desired format or perform other processing here

        return jsonify({'prediction': prediction.tolist()})  # Return the prediction results as JSON
    except Exception as e:
        return str(e), 400  # Return error message and status code 400 for bad requests

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Run the Flask app

