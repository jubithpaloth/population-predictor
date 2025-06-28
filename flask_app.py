
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import pickle
import io
import json
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model on startup
model_data = None

def load_model():
    global model_data
    try:
        with open('advanced_str_population_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

def predict_profile(profile_data):
    """Make prediction for a single profile"""
    if not model_data:
        return None

    # Prepare input data
    str_markers = model_data['str_markers']
    input_vector = []

    for marker in str_markers:
        if marker in profile_data:
            input_vector.append(profile_data[marker])
        else:
            # Use population mean for missing markers
            pop_means = model_data['population_centroids']
            overall_mean = np.mean([pop_means[pop][marker] for pop in pop_means.keys() 
                                  if marker in pop_means[pop]])
            input_vector.append(overall_mean)

    input_array = np.array(input_vector).reshape(1, -1)

    # Get predictions from ensemble
    ensemble_model = model_data['ensemble_model']
    prediction = ensemble_model.predict(input_array)[0]
    probabilities = ensemble_model.predict_proba(input_array)[0]

    # Get class labels
    classes = ensemble_model.classes_

    # Create results with top predictions
    results = []
    prob_pairs = list(zip(classes, probabilities))
    prob_pairs.sort(key=lambda x: x[1], reverse=True)

    for i, (pop, prob) in enumerate(prob_pairs[:5]):
        results.append({
            'rank': i + 1,
            'population': pop,
            'confidence': round(prob * 100, 2)
        })

    return results

@app.route('/')
def index():
    if not model_data:
        return "Model not loaded. Please check server logs."

    str_markers = model_data['str_markers']
    return render_template('index.html', str_markers=str_markers)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        profile_data = {}

        for marker, value in data.items():
            if value and float(value) > 0:
                profile_data[marker] = float(value)

        if not profile_data:
            return jsonify({'error': 'Please enter at least one STR marker value'})

        results = predict_profile(profile_data)

        if results:
            return jsonify({'success': True, 'results': results})
        else:
            return jsonify({'error': 'Prediction failed'})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})

        if file and file.filename.endswith('.csv'):
            # Read CSV file
            df = pd.read_csv(file)

            # Process each row
            str_markers = model_data['str_markers']
            available_markers = [col for col in df.columns if col in str_markers]

            if not available_markers:
                return jsonify({'error': 'No STR markers found in the uploaded file'})

            all_results = []

            for idx, row in df.iterrows():
                profile_data = {}
                for marker in available_markers:
                    if pd.notna(row[marker]):
                        profile_data[marker] = float(row[marker])

                if profile_data:
                    results = predict_profile(profile_data)
                    if results:
                        all_results.append({
                            'sample_id': f"Sample_{idx+1}",
                            'top_population': results[0]['population'],
                            'confidence': results[0]['confidence'],
                            'all_predictions': results
                        })

            return jsonify({
                'success': True, 
                'results': all_results,
                'markers_found': available_markers
            })

        else:
            return jsonify({'error': 'Please upload a CSV file'})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/template')
def download_template():
    try:
        str_markers = model_data['str_markers']
        template_data = {marker: [0.0] for marker in str_markers}
        template_df = pd.DataFrame(template_data)

        output = io.StringIO()
        template_df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='str_template.csv'
        )

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
