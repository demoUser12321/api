from flask import Flask, request, jsonify
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import json
import firebase_admin
from firebase_admin import credentials, storage
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Initialize Firebase Admin SDK
cred_path = 'placementprediction-fp-firebase-adminsdk-miab4-e87b43e44b.json'
cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'placementprediction-fp.appspot.com'
})

class LSTMModel:
    def __init__(self, look_back=4):
        self.look_back = look_back
        self.model = Sequential([
            LSTM(50, input_shape=(look_back, 1)),
            Dense(1)
        ])
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def fit(self, X, Y, epochs=100, batch_size=1, verbose=2):
        self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def predict(self, data):
        return self.model.predict(data)

def preprocess_data(company_data):
    df_skills = company_data[company_data['Placed'] == 1].copy()
    df_skills['SkillCount'] = df_skills.groupby(['Year', 'Skills'])['Skills'].transform('count')

    df_counts = df_skills[['Year', 'SkillCount']].groupby('Year').sum().rename(columns={'SkillCount': 'TotalCount'}).reset_index()
    df_skills = df_skills.drop_duplicates(subset=['Year', 'Skills']).reset_index(drop=True)
    df_skills['SkillPercentage'] = (df_skills['SkillCount'] / df_counts.set_index('Year').loc[df_skills['Year']]['TotalCount'].values) * 100

    return df_counts, df_skills

def prepare_datasets(df_counts, look_back, scaler):
    dataset = scaler.fit_transform(df_counts['TotalCount'].values.reshape(-1, 1))
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    X = np.array(X).reshape(-1, look_back, 1)
    Y = np.array(Y)
    return X, Y

def make_predictions(model, df_counts, df_skills, scaler, look_back):
    predictions = {}
    dataset = scaler.transform(df_counts['TotalCount'].values.reshape(-1, 1))
    for i in range(look_back, len(df_counts)):
        last_data = np.array([dataset[i-look_back:i, 0]]).reshape(1, look_back, 1)
        prediction = model.predict(last_data)
        predicted_count = scaler.inverse_transform(prediction).flatten()[0]
        year = df_counts['Year'].iloc[i]

        skills_data = df_skills[df_skills['Year'] == year]
        skill_dict = dict(zip(skills_data['Skills'], skills_data['SkillPercentage'].round(2)))

        predictions[str(year)] = {"skills": skill_dict, "count": int(round(predicted_count))}
    return predictions

def train_predict_company(company_data, look_back=4):
    lstm_model = LSTMModel(look_back)
    df_counts, df_skills = preprocess_data(company_data)
    X, Y = prepare_datasets(df_counts, look_back, lstm_model.scaler)
    lstm_model.fit(X, Y)
    return make_predictions(lstm_model, df_counts, df_skills, lstm_model.scaler, look_back)

def generate_lstm_report(companies_data):
    full_predictions = {}
    for company, company_data in companies_data.items():
        print(f"Processing predictions for {company}...")
        predictions = train_predict_company(company_data)
        full_predictions[company] = predictions

    # JSON output
    json_data = json.dumps(full_predictions, indent=4)

    # Upload JSON file to Firebase Storage
    bucket = storage.bucket()
    blob = bucket.blob('lstm_prediction_report.json')
    blob.upload_from_string(json_data, content_type='application/json')

    return 'lstm_prediction_report.json'

def fetch_file_stream(file_path):
    bucket = storage.bucket()
    blob = bucket.blob(file_path)

    byte_stream = BytesIO()
    blob.download_to_file(byte_stream)
    byte_stream.seek(0)

    return byte_stream

def load_data_from_firebase(file_path, skip_header=False):
    stream = fetch_file_stream(file_path)
    if skip_header:
        return pd.read_csv(stream, skiprows=1).dropna()
    else:
        return pd.read_csv(stream).dropna()

def merge_and_upload_to_firebase(data, new_file_path):
    new_data = load_data_from_firebase(new_file_path, skip_header=True)
    if new_data.empty:
        print("Warning: 'newfile.csv' is empty. No data will be merged.")
        return 'data.csv'

    merged_data = pd.concat([data, new_data], ignore_index=True)
    csv_data = merged_data.to_csv(index=False)

    bucket = storage.bucket()
    blob = bucket.blob('data.csv')
    blob.upload_from_string(csv_data, content_type='text/csv')

    return 'data.csv'

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file temporarily
    file.filename = 'newfile.csv'
    bucket = storage.bucket()
    # Copy file from cache directory to Firebase Storage
    blob = bucket.blob(file.filename)
    blob.upload_from_file(file, content_type='text/csv')

    # Trigger the Python script
    firebase_file_path = 'data.csv'
    data = load_data_from_firebase(firebase_file_path)
    data_exploded = data.explode('CompanyName')
    data_exploded.rename(columns={'CompanyName': 'Company'}, inplace=True)

    merged_data = merge_and_upload_to_firebase(data_exploded, file.filename)
    companies_data = {company: data_exploded[data_exploded['Company'] == company].drop(columns=['Company']) for company in data_exploded['Company'].unique()}

    generated_file = generate_lstm_report(companies_data)
    return jsonify({"message": f"LSTM Predictions saved to {generated_file}"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
