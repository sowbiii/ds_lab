from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import preprocess_data
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
app = Flask(__name__)

# Homepage Route
@app.route('/')
def index():
    return render_template('index.html')

# Route for displaying results
@app.route('/results', methods=['POST'])
def results():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    # Load and preprocess data
    data = pd.read_csv(file)
    processed_data, model_results, metrics, best_params = preprocess_data(data)

    # Pass results to template
    return render_template('results.html', metrics=metrics, best_params=best_params, results=model_results)

if __name__ == '__main__':
    app.run(debug=True)
