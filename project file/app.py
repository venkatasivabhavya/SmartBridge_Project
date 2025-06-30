from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open(r"C:\Users\venka\OneDrive\Desktop\project file\rf_acc_68.pkl","rb"))
scaler = pickle.load(open(r"C:\Users\venka\OneDrive\Desktop\project file\normalizer.pkl","rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_input = scaler.transform([features])
        prediction = model.predict(final_input)

        result = "High Risk of Liver Cirrhosis" if prediction[0] == 1 else "Low Risk of Liver Cirrhosis"
        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)