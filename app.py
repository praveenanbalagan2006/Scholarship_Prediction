from flask import Flask, request, render_template  # type: ignore
import pickle
import numpy as np  # type: ignore

app = Flask(__name__)

# Load trained model, encoder, and feature names
model = pickle.load(open("scholarship_modelnew.pkl", "rb"))
encoder = pickle.load(open("encodernew.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))
scholarship_names = ['INSPIRE Scholarship are the Scholarship for Higher Education (SHE)','Abdul Kalam Technology Innovation National Fellowship','AAI Sports Scholarship Scheme in India ', 'Glow and lovely Career Foundation Scholarship','National Fellowship for Persons with Disabilities','ONGC Sports Scholarship Scheme ','Pragati Scholarship ? AICTE-Scholarship Scheme to Girl Child','Dr. Ambedkar post matric Scholarship','Indira Gandhi Scholarship for Single Girl Child UGC Scholarship for PG Programmes','National Overseas Scholarship Scheme 2021-22']

# Define categorical feature order
categorical_columns = [
    'Education Qualification', 'Gender', 'Community', 'Religion', 'Exservice-men', 
    'Disability', 'Sports', 'Annual-Percentage', 'Income', 'India'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract user input including Name
        user_name = request.form.get("Name", "User")
        input_data = [request.form.get(col, "") for col in categorical_columns]
        
        # Convert to numpy array
        input_data = np.array([input_data])

        # Encode input features
        input_encoded = encoder.transform(input_data)

        # Debugging prints
        print("🔹 User Name:", user_name)
        print("🔹 User Input:", input_data)
        print("🔹 Encoded Features Shape:", input_encoded.shape)
        print("🔹 Encoded Feature Values:", input_encoded)

        # Predict outcome
        prediction = model.predict(input_encoded)
        predicted_scholarship = scholarship_names[prediction[0]]  # Map prediction index to scholarship name

        result = f"✅ {user_name}, you are eligible for: **{predicted_scholarship}** 🎉" 

        return render_template('result.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
