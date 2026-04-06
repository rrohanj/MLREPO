from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
# Ensure these files are in the same directory as app.py
model = joblib.load('water_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # These MUST match the 'name' attributes in your HTML exactly (all lowercase)
        feature_order = [
            'ph', 'hardness', 'solids', 'chloramines', 'sulfate', 
            'cond', 'carb', 'tri', 'turb'
        ]
        
        # Extract values using the exact keys from the HTML form
        input_data = []
        for name in feature_order:
            val = request.form.get(name)
            if val is not None and val != "":
                input_data.append(float(val))
            else:
                input_data.append(0.0) # Default for empty fields
        
        # Process for model
        final_features = np.array(input_data).reshape(1, -1)
        scaled_features = scaler.transform(final_features)
        
        # Get Prediction and Probability
        prediction = model.predict(scaled_features)[0]
        # XGBoost proba returns [P(Unsafe), P(Safe)]
        probability = model.predict_proba(scaled_features)[0][1] * 100 
        
        status = "Potable (Safe)" if prediction == 1 else "Not Potable (Unsafe)"
        color = "#2ecc71" if prediction == 1 else "#e74c3c"
        
        # Return to template - ensure variable names match the {{ }} in index.html
        return render_template('index.html', 
                               prediction_text=status, 
                               prob=round(probability, 2),
                               res_color=color,
                               input_values=input_data)
                               
    except Exception as e:
        print(f"Error: {e}") # This will show in your terminal/CMD
        return render_template('index.html', prediction_text="Error: Check Inputs")

if __name__ == "__main__":
    app.run(debug=True)