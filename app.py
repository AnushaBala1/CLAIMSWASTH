from flask import Flask, flash, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from config import Config
from datetime import datetime
import os
from werkzeug.utils import secure_filename
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import re

# Load your model and preprocessing tools
model = joblib.load('model.pkl')
scaler = joblib.load('scaler (1).pkl')
label_encoders = joblib.load('label_encoders (1).pkl')


app = Flask(__name__)
app.config.from_object(Config)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy and Bcrypt
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Optional: Configure maximum file size (in bytes)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

'''# Load preprocessing tools
#imputer = joblib.load('imputer.pkl')
label_encoders = joblib.load('label_encoders(1).pkl')
scaler = joblib.load('scaler(1).pkl')
model = joblib.load('model.pkl')'''



# Define the User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False, unique=True)

# Define the Agent model
class Agent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False, unique=True)

# Route for the homepage (login page)
@app.route('/')
def home():
    return render_template('frontpage.html')

# Route for user login
@app.route('/user_login', methods=['POST'])
def user_login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    user = User.query.filter_by(username=username).first()

    if not user:
        return jsonify({"error": "username", "message": "Username does not exist"}), 400
    elif not bcrypt.check_password_hash(user.password, password):
        return jsonify({"error": "password", "message": "Incorrect password"}), 400
    
    session['user_id'] = user.id
    return redirect(url_for('user_dashboard'))

# Route for agent login
@app.route('/agent_login', methods=['POST'])
def agent_login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    agent = Agent.query.filter_by(username=username).first()

    if not agent:
        return jsonify({"error": "username", "message": "Username does not exist"}), 400
    elif not bcrypt.check_password_hash(agent.password, password):
        return jsonify({"error": "password", "message": "Incorrect password"}), 400

    session['agent_id'] = agent.id
    return redirect(url_for('agent_dashboard'))

@app.route('/agent_logout',methods=['POST'])
def agent_logout():
    if 'agent_id' in session:
        session.pop('agent_id',None)
        return redirect(url_for('home'))

@app.route('/user_dashboard')
def user_dashboard():
    if 'user_id' in session:
        return render_template('dashboard.html')  
    else:
        return redirect(url_for('home'))

@app.route('/need_to_submit_claim', methods=['POST'])
def need_to_submit_claim():
    return render_template('submit_claim.html')
  # Your new page template

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension."""
    allowed_extensions = {'pdf', 'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def extract_text_from_document(file_path):
    """
    Extract text from a PDF or image file using OCR.
    """
    extracted_text = ""
    try:
        if file_path.lower().endswith('.pdf'):
            pages = convert_from_path(file_path, dpi=300)
            for page in pages:
                extracted_text += pytesseract.image_to_string(page, lang="eng") + "\n"
            print(f"OCR successfully executed for: {file_path}")
        else:
            image = Image.open(file_path)
            extracted_text = pytesseract.image_to_string(image, lang="eng")
            print(f"OCR successfully executed for: {file_path}")
    except Exception as e:
        print(f"Error during OCR for file {file_path}: {e}")
    return extracted_text

'''import pdfplumber
from datetime import datetime
import re

def extract_text_from_document(file_path):
    """
    Extract text from a PDF file using pdfplumber.
    """
    extracted_text = ""
    try:
        if file_path.lower().endswith('.pdf'):
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    extracted_text += page.extract_text() + "\n"  # Extract text from each page
            print(f"Text extraction successfully executed for: {file_path}")
        else:
            extracted_text = ""  # For non-PDF files, no text extraction will be performed
            print(f"Unsupported file type: {file_path}")
    except Exception as e:
        print(f"Error during text extraction for file {file_path}: {e}")
    
    return extracted_text'''

'''def validate_field(user_value, extracted_text, field_type="text"):
    print(f"Validating: {user_value} in {extracted_text}")
    if not user_value:
        return 0

    if field_type == "date":
        # Normalize user-entered date to YYYY-MM-DD format
        try:
            user_date = datetime.strptime(user_value, '%Y-%m-%d').date()
            print(f"User date: {user_date}")
        except ValueError:
            return 0

        # Extract and normalize dates from the text
        date_patterns = [
            r"(\d{2}/\d{2}/\d{4})",  # Matches DD/MM/YYYY
            r"(\d{4}-\d{2}-\d{2})"   # Matches YYYY-MM-DD
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, extracted_text)
            for match in matches:
                print(f"Match found: {match}")
                try:
                    # Normalize the matched date to YYYY-MM-DD
                    extracted_date = datetime.strptime(match, '%d/%m/%Y').date() if '/' in match else datetime.strptime(match, '%Y-%m-%d').date()
                    print(f"Extracted date: {extracted_date}")
                    if extracted_date == user_date:
                        print(f"Match found: {extracted_date} == {user_date}")
                        return 1  # Valid match
                    
                except ValueError:
                    continue

        return 0  # No valid match found

    # Default text-based validation (non-date fields)
    return 1 if str(user_value).lower() in extracted_text.lower() else 0'''

from datetime import datetime
import re

def validate_field(user_value, extracted_text, field_type="text"):
    """
    Validate user-entered data against extracted text.

    Args:
        user_value (str): The value entered by the user.
        extracted_text (str): Text extracted from the document.
        field_type (str): Type of field being validated ("text" or "date").

    Returns:
        int: 1 if valid, 0 if invalid.
    """
    if not user_value:
        return 0

    if field_type == "date":
        # Normalize user-entered date to YYYY-MM-DD format
        try:
            user_date = datetime.strptime(user_value, '%Y-%m-%d').date()
        except ValueError:
            return 0

        # Extract and normalize dates from the text
        date_patterns = [
    r"(\d{2}/\d{2}/\d{4})",  # Matches DD/MM/YYYY
    r"(\d{4}-\d{2}-\d{2})"   # Matches YYYY-MM-DD
    ]

        for pattern in date_patterns:
            matches = re.findall(pattern, extracted_text)
            for match in matches:
                try:
                    # Normalize the matched date to YYYY-MM-DD
                    extracted_date = datetime.strptime(match, '%d/%m/%Y').date() if '/' in match else datetime.strptime(match, '%Y-%m-%d').date()
                    if extracted_date == user_date:
                        return 1  # Valid match
                except ValueError:
                    continue

        return 0  # No valid match found

    # Default text-based validation (non-date fields)
    return 1 if str(user_value).lower() in extracted_text.lower() else 0

class Claim(db.Model):
    __tablename__ = 'claims'

    id = db.Column(db.Integer, primary_key=True)
    claim_type = db.Column(db.String(50), nullable=False)
    claim_date = db.Column(db.Date, nullable=False)
    policy_type=db.Column(db.String(50), nullable=False)
    policy_coverage=db.Column(db.Float, nullable=False)
    policy_start_date=db.Column(db.Date, nullable=False)
    policy_end_date=db.Column(db.Date, nullable=False)
    proposer_name = db.Column(db.String(100), nullable=False)
    customer_id = db.Column(db.String(100), nullable=True)
    patient_name = db.Column(db.String(100), nullable=False)
    patient_gender = db.Column(db.String(10), nullable=False)
    patient_age = db.Column(db.Float, nullable=False)
    patient_relationship = db.Column(db.String(50), nullable=False)
    accident_date = db.Column(db.Date, nullable=True)
    accident_time = db.Column(db.Time, nullable=True)
    reported_to_police = db.Column(db.String(10), nullable=False)
    previous_claims = db.Column(db.String(10), nullable=False)
    diagnosis = db.Column(db.String(200), nullable=False)
    procedure_type = db.Column(db.String(50), nullable=False)
    admission_date = db.Column(db.Date, nullable=False)
    discharge_date = db.Column(db.Date, nullable=False)
    admission_type = db.Column(db.String(50), nullable=False)
    hospitalization_expenses = db.Column(db.Float, nullable=False)
    pre_hospitalization_expenses = db.Column(db.Float, nullable=True)
    post_hospitalization_expenses = db.Column(db.Float, nullable=True)
    ambulance_charges = db.Column(db.Float, nullable=True)
    other_expenses = db.Column(db.Float, nullable=True)
    Amount_Claimed = db.Column(db.Float, nullable=True)
    hospital_name = db.Column(db.String(200), nullable=False)
    hospital_city = db.Column(db.String(100), nullable=False)
    hospital_type = db.Column(db.String(50), nullable=False)
    treating_doctor_name = db.Column(db.String(100), nullable=False)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), nullable=False, default="Pending")
    check_patient_name = db.Column(db.Integer, default=0)
    check_admission_date = db.Column(db.Integer, default=0)
    check_discharge_date = db.Column(db.Integer, default=0)
    check_doctor_name = db.Column(db.Integer, default=0)
    check_Amount_Claimed = db.Column(db.Integer,default=0)    
    final_bill = db.Column(db.String(200), nullable=True)
    diagnostic_reports = db.Column(db.String(200), nullable=True)
    prescriptions = db.Column(db.String(200), nullable=True)
    accident_report = db.Column(db.String(200), nullable=True)


@app.route('/submit_claim', methods=['POST'])
def submit_claim():
    # Get data from the form
    claim_type = request.form.get('claim_type')
    claim_date = request.form.get('claim_date')
    policy_type= request.form.get('claim_date')
    policy_coverage= request.form.get('policy_coverage')
    policy_start_date= request.form.get('policy_start_date')
    policy_end_date= request.form.get('policy_end_date')
    proposer_name = request.form.get('proposer_name')
    customer_id = request.form.get('customer_id')
    patient_name = request.form.get('patient_name')
    patient_gender = request.form.get('patient_gender')
    patient_age = request.form.get('patient_age')
    patient_relationship = request.form.get('patient_relationship')
    accident_date = request.form.get('accident_date')
    accident_time = request.form.get('accident_time')
    reported_to_police = request.form.get('reported_to_police')
    previous_claims = request.form.get('previous_claims')
    diagnosis = request.form.get('diagnosis')
    procedure_type = request.form.get('procedure_type')
    admission_date = request.form.get('admission_date')
    discharge_date = request.form.get('discharge_date')
    admission_type = request.form.get('admission_type')
    hospitalization_expenses = request.form.get('hospitalization_expenses')
    pre_hospitalization_expenses = request.form.get('pre_hospitalization_expenses')
    post_hospitalization_expenses = request.form.get('post_hospitalization_expenses')
    ambulance_charges = request.form.get('ambulance_charges')
    other_expenses = request.form.get('other_expenses')
    Amount_Claimed = request.form.get('Amount_Claimed')
    hospital_name = request.form.get('hospital_name')
    hospital_city = request.form.get('hospital_city')
    hospital_type = request.form.get('hospital_type')
    treating_doctor_name = request.form.get('treating_doctor_name')

    final_bill = request.files['final_bill']
    diagnostic_reports = request.files['diagnostic_reports']
    prescriptions = request.files['prescriptions']
    accident_report = request.files.get('accident_report')  # Optional file

    # Save file paths
    '''final_bill_filename = secure_filename(final_bill.filename)
    final_bill.save(os.path.join(app.config['UPLOAD_FOLDER'], final_bill_filename))
    '''
    if final_bill and allowed_file(final_bill.filename):
        # Secure the filename and save the file
        final_bill_filename = secure_filename(final_bill.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], final_bill_filename)
        final_bill.save(file_path)
        # Perform OCR on the final bill
        extracted_text = extract_text_from_document(file_path)
    print(f"Extracted Text: {extracted_text}")
    diagnostic_reports_filename = secure_filename(diagnostic_reports.filename)
    diagnostic_reports.save(os.path.join(app.config['UPLOAD_FOLDER'], diagnostic_reports_filename))

    prescriptions_filename = secure_filename(prescriptions.filename)
    prescriptions.save(os.path.join(app.config['UPLOAD_FOLDER'], prescriptions_filename))

    accident_report_filename = None
    if accident_report:
        accident_report_filename = secure_filename(accident_report.filename)
        accident_report.save(os.path.join(app.config['UPLOAD_FOLDER'], accident_report_filename))
    
    check_patient_name = 1 if validate_field(patient_name, extracted_text) else 0
    check_admission_date = 1 if validate_field(admission_date, extracted_text, field_type="date") else 0
    check_discharge_date = 1 if validate_field(discharge_date, extracted_text, field_type="date") else 0
    check_doctor_name = 1 if validate_field(treating_doctor_name, extracted_text) else 0
    check_Amount_Claimed = 1 if validate_field(Amount_Claimed, extracted_text) else 0
    
    # Create a new Claim object
    new_claim = Claim(
        claim_type=claim_type,
        claim_date=datetime.strptime(claim_date, '%Y-%m-%d'),
        policy_type=policy_type,
        policy_coverage=policy_coverage,
        policy_start_date=datetime.strptime(policy_start_date, '%Y-%m-%d'),
        policy_end_date=datetime.strptime(policy_end_date, '%Y-%m-%d'),
        proposer_name=proposer_name,
        customer_id=customer_id,
        patient_name=patient_name,
        patient_gender=patient_gender,
        patient_age=patient_age,
        patient_relationship=patient_relationship,
        accident_date=datetime.strptime(accident_date, '%Y-%m-%d') if accident_date else None,
        accident_time=datetime.strptime(accident_time, '%H:%M') if accident_time else None,
        reported_to_police=reported_to_police,
        previous_claims=previous_claims,
        diagnosis=diagnosis,
        procedure_type=procedure_type,
        admission_date=datetime.strptime(admission_date, '%Y-%m-%d'),
        discharge_date=datetime.strptime(discharge_date, '%Y-%m-%d'),
        admission_type=admission_type,
        hospitalization_expenses=float(hospitalization_expenses),
        pre_hospitalization_expenses=float(pre_hospitalization_expenses) if pre_hospitalization_expenses else 0,
        post_hospitalization_expenses=float(post_hospitalization_expenses) if post_hospitalization_expenses else 0,
        ambulance_charges=float(ambulance_charges) if ambulance_charges else 0,
        other_expenses=float(other_expenses) if other_expenses else 0,
        Amount_Claimed=float(Amount_Claimed) if Amount_Claimed else 0,
        hospital_name=hospital_name,
        hospital_city=hospital_city,
        hospital_type=hospital_type,
        treating_doctor_name=treating_doctor_name,

        final_bill = final_bill_filename,
        diagnostic_reports = diagnostic_reports_filename,
        prescriptions = prescriptions_filename,
        accident_report = accident_report_filename,
        check_patient_name = check_patient_name,
        check_admission_date = check_admission_date,
        check_discharge_date = check_discharge_date,
        check_doctor_name = check_doctor_name,
        check_Amount_Claimed = check_Amount_Claimed
    )

    # Add to session and commit to the database
    db.session.add(new_claim)
    db.session.commit()

    # redirect to confirmation page
    return redirect(url_for('claim_submitted', claim_id=new_claim.id))


@app.route('/claim_submitted')
def claim_submitted():
    claim_id = request.args.get('claim_id', type=int)
    claim = Claim.query.get_or_404(claim_id) 
    return render_template('claim_submitted.html',claim=claim)

# Correct the database retrieval part
#claims_with_null_status = Claim.query.filter(Claim.status == 'Pending').all()

# Then convert to DataFrame
#user_df = pd.DataFrame([(claim.id, claim.claim_type, claim.claim_date) for claim in claims_with_null_status], 
#                       columns=["id", "claim_type", "claim_date"])  # Select the columns you need

#@app.route('/predict_claim_status', methods=['GET'])
'''def predict_claim_status1():
    # Retrieve claims with 'Pending' status from the database
    claims_with_null_status =Claim.query.filter(Claim.status == 'Pending').all()

# Assuming claims_with_null_status contains the claim objects
    user_df = pd.DataFrame([(claim.claim_date, claim.Amount_Claimed,
                          claim.patient_gender,claim.policy_start_date,claim.policy_end_date,
                         claim.patient_age, claim.diagnosis, claim.admission_date, claim.discharge_date,
                         claim.hospitalization_expenses, claim.pre_hospitalization_expenses, claim.post_hospitalization_expenses,
                         claim.ambulance_charges, claim.other_expenses, claim.policy_type, claim.policy_coverage,claim.check_patient_name,
                         claim.check_admission_date,claim.check_discharge_date,claim.check_doctor_name,claim.check_Amount_Claimed)
                        for claim in claims_with_null_status],
                       columns=[ "claim_date", "Amount_Claimed", 
                                 "patient_gender", "age", "diagnosis","policy_start_date","policy_end_date",
                                 "admission_date", "discharge_date", "hospitalization_expenses",
                                "pre_hospitalization_expenses", "post_hospitalization_expenses", "ambulance_charges",
                                "other_expenses", "policy_type", "policy_coverage","check_patient_name","check_admission_date",
                                "check_discharge_date","check_doctor_name","check_Amount_Claimed"])

    # Load necessary components for prediction
    training_columns = joblib.load('training_columns (1).pkl')  # Loaded the training column order
    numerical_imputer = joblib.load('numerical_imputer (1).pkl')  # Imputer for numerical features
    categorical_imputer = joblib.load('categorical_imputer copy.pkl')  # Imputer for categorical features
    label_encoders = joblib.load('label_encoders (1).pkl')  # Label encoders for categorical columns
    scaler = joblib.load('scaler (1).pkl')  # Scaler for numerical features
    model = joblib.load('model.pkl')  # Trained model
    target_encoder = joblib.load('target_encoder (2).pkl')  # Encoder for target labels

    # Preprocess the DataFrame to match training data
    user_df = user_df.reindex(columns=training_columns, fill_value=0)  # Ensure columns match training

    # Convert date columns to timestamp (since datetime columns need to be transformed as integers)
    datetime_cols = ['Claim_Date', 'Date_of_Admission', 'Date_of_Discharge']
    for col in datetime_cols:
        if col in user_df.columns:
            user_df[col] = pd.to_datetime(user_df[col], errors='coerce')
            user_df[col] = ((user_df[col] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')).astype('int64')  # Convert to days since epoch

    # Impute missing numerical and categorical values using the saved imputers
    numerical_columns = numerical_imputer.feature_names_in_  # Get the names of numerical columns
    categorical_columns = categorical_imputer.feature_names_in_  # Get the names of categorical columns
    user_df[numerical_columns] = numerical_imputer.transform(user_df[numerical_columns])  # Impute numerical columns
    user_df[categorical_columns] = categorical_imputer.transform(user_df[categorical_columns])  # Impute categorical columns

    # Encode categorical columns using label encoders
    for column in categorical_columns:
        user_df[column] = label_encoders[column].transform(user_df[column])  # Transform categorical features

    # Scale the numerical columns
    user_df[numerical_columns] = scaler.transform(user_df[numerical_columns])  # Scale numerical features

    # Predict the claim status
    predictions = model.predict(user_df)  # Model prediction

    # Decode the prediction to the original target labels
    result = target_encoder.inverse_transform(predictions)  # Convert the numeric prediction to its label

    # Update the claim statuses in the database
    for i, claim in enumerate(claims_with_null_status):
        if i < len(result):  # Ensure we don't exceed the number of predictions
            claim.status = result[i]  # Update the claim status with the predicted value

    # Commit the changes to the database
    db.session.commit()
    return
    # Return a success message
    #return jsonify({"message": "Claim statuses updated successfully"})

def predict_claim_status():
    # Load components
    training_columns = joblib.load('training_columns.pkl')
    numerical_imputer = joblib.load('numerical_imputer.pkl')
    categorical_imputer = joblib.load('categorical_imputer.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('model.pkl')
    target_encoder = joblib.load('target_encoder.pkl')

    # Print the known classes
    print("Known classes in target encoder:", target_encoder.classes_)

    # Retrieve pending claims
    claims_with_null_status = Claim.query.filter(Claim.status == 'Pending').all()

    # Create DataFrame
    user_df = pd.DataFrame([(
        claim.claim_date,
        claim.Amount_Claimed,
        claim.patient_gender,
        claim.policy_start_date,
        claim.policy_end_date,
        claim.patient_age,
        claim.diagnosis,
        claim.admission_date,
        claim.discharge_date,
        claim.hospitalization_expenses,
        claim.pre_hospitalization_expenses,
        claim.post_hospitalization_expenses,
        claim.ambulance_charges,
        claim.other_expenses,
        claim.policy_type,
        claim.policy_coverage
    ) for claim in claims_with_null_status],
    columns=[
        "claim_date",
        "Amount_Claimed",
        "patient_gender",
        "policy_start_date",
        "policy_end_date",
        "age",
        "diagnosis",
        "admission_date",
        "discharge_date",
        "hospitalization_expenses",
        "pre_hospitalization_expenses",
        "post_hospitalization_expenses",
        "ambulance_charges",
        "other_expenses",
        "policy_type",
        "policy_coverage"
    ])

    try:
        # Ensure columns match training data
        user_df = user_df.reindex(columns=training_columns, fill_value=0)

        # Handle datetime columns
        date_columns = ['claim_date', 'admission_date', 'discharge_date', 'policy_start_date', 'policy_end_date']
        for col in date_columns:
            if col in user_df.columns:
                user_df[col] = pd.to_datetime(user_df[col])
                user_df[col] = ((user_df[col] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')).astype('int64')

        # Get numerical and categorical columns
        numerical_columns = numerical_imputer.feature_names_in_
        categorical_columns = categorical_imputer.feature_names_in_

        # Impute missing values
        user_df[numerical_columns] = numerical_imputer.transform(user_df[numerical_columns])
        user_df[categorical_columns] = categorical_imputer.transform(user_df[categorical_columns])

        # Encode categorical features
        for column in categorical_columns:
            # Handle unseen categories
            unique_values = user_df[column].unique()
            known_categories = label_encoders[column].classes_
            unknown_categories = set(unique_values) - set(known_categories)
            
            if unknown_categories:
                print(f"Warning: New categories found in {column}: {unknown_categories}")
                # Map unknown categories to a known category or handle appropriately
                user_df[column] = user_df[column].map(lambda x: known_categories[0] if x in unknown_categories else x)
            
            user_df[column] = label_encoders[column].transform(user_df[column])

        # Scale numerical features
        user_df[numerical_columns] = scaler.transform(user_df[numerical_columns])

        # Make predictions
        predictions = model.predict(user_df)
        
        # Convert predictions to original labels
        result = target_encoder.inverse_transform(predictions)

        # Update database
        for claim, status in zip(claims_with_null_status, result):
            claim.status = status
        
        db.session.commit()
        
        return True

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return False'''

def predict_claim_status():
    # Load pre-trained components
    training_columns = joblib.load('training_columns.pkl')
    numerical_imputer = joblib.load('numerical_imputer.pkl')
    categorical_imputer = joblib.load('categorical_imputer.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('model.pkl')
    target_encoder = joblib.load('target_encoder.pkl')

    # Retrieve claims with 'Pending' status from the database
    claims_with_null_status = Claim.query.filter(Claim.status == 'Pending').all()

    # Create DataFrame for prediction
    user_df = pd.DataFrame([(
        claim.claim_date,
        claim.Amount_Claimed,
        claim.patient_gender,
        claim.policy_start_date,
        claim.policy_end_date,
        claim.patient_age,
        claim.diagnosis,
        claim.admission_date,
        claim.discharge_date,
        claim.hospitalization_expenses,
        claim.pre_hospitalization_expenses,
        claim.post_hospitalization_expenses,
        claim.ambulance_charges,
        claim.other_expenses,
        claim.policy_type,
        claim.policy_coverage,
        claim.check_patient_name,
        claim.check_admission_date,
        claim.check_discharge_date,
        claim.check_Amount_Claimed,
        claim.check_doctor_name
    ) for claim in claims_with_null_status],
    columns=[
        "claim_date",
        "Amount_Claimed",
        "patient_gender",
        "policy_start_date",
        "policy_end_date",
        "age",
        "diagnosis",
        "admission_date",
        "discharge_date",
        "hospitalization_expenses",
        "pre_hospitalization_expenses",
        "post_hospitalization_expenses",
        "ambulance_charges",
        "other_expenses",
        "policy_type",
        "policy_coverage",
        "check_patient_name",
        "check_admission_date",
        "check_discharge_date",
        "check_Amount_Claimed",
        "check_doctor_name"
    ])

    try:
        # Validate binary columns
        binary_cols = ['check_patient_name', 'check_admission_date', 'check_discharge_date', 'check_Amount_Claimed', 'check_doctor_name']
        if (user_df[binary_cols] == 0).any(axis=1).any():
            print("Claim rejected due to invalid values in check columns.")
            for claim in claims_with_null_status:
                claim.status = "Rejected"
            db.session.commit()
            return "Claims rejected due to invalid checks."

        # Reindex DataFrame to match training columns
        user_df = user_df.reindex(columns=training_columns, fill_value=0)

        # Handle datetime columns
        date_columns = ['claim_date', 'admission_date', 'discharge_date', 'policy_start_date', 'policy_end_date']
        for col in date_columns:
            if col in user_df.columns:
                user_df[col] = pd.to_datetime(user_df[col], errors='coerce')
                user_df[col] = ((user_df[col] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1D')).astype('int64')

        # Impute missing values
        numerical_columns = numerical_imputer.feature_names_in_
        categorical_columns = categorical_imputer.feature_names_in_
        user_df[numerical_columns] = numerical_imputer.transform(user_df[numerical_columns])
        user_df[categorical_columns] = categorical_imputer.transform(user_df[categorical_columns])

        # Encode categorical columns
        for column in categorical_columns:
            user_df[column] = label_encoders[column].transform(user_df[column])

        # Scale numerical features
        user_df[numerical_columns] = scaler.transform(user_df[numerical_columns])

        # Predict claim status
        predictions = model.predict(user_df)

        # Decode predictions
        result = target_encoder.inverse_transform(predictions)

        # Update database with predictions
        for claim, status in zip(claims_with_null_status, result):
            claim.status = status
        db.session.commit()

        return "Claim statuses updated successfully."

    except Exception as e:
        print(f"Error in claim status prediction: {str(e)}")
        return "Error in claim status prediction."



@app.route('/agent_dashboard')
def agent_dashboard():
    if 'agent_id' in session:  
        # Fetch all claims from the database
        claims = Claim.query.order_by(Claim.claim_date.desc()).all()

        total_claims = len(claims)
        claims_approved = len([claim for claim in claims if claim.status == "Approved"])
        claims_rejected = len([claim for claim in claims if claim.status == "Rejected"])

        # Render the template with claims and summary data
        return render_template(
            'agent_dashboard.html', 
            claims=claims,
            total_claims=total_claims,
            claims_approved=claims_approved,
            claims_rejected=claims_rejected
        )
    else:
        return redirect(url_for('home'))  # Redirect if the agent isn't logged in

@app.route('/validate_claim/<int:claim_id>')
def validate_claim(claim_id):
    claim = Claim.query.get_or_404(claim_id)
    predict_claim_status()
    
    # Mock Model Prediction and Document Validation
    recommendation = {
        'probability': 87,
        'action': "Likely to Approve",
        'documents': {
            'final_bill_status': "Valid",
            'prescriptions_status': "Valid",
            'diagnostic_reports_status': "Valid",
            'accident_report_status': "Missing"
        }
    }

    # Store recommendation in session
    session[f"recommendation_{claim_id}"] = recommendation

    # Render final.html with evaluation details
    return render_template('final.html', claim=claim, evaluation=recommendation)


@app.route('/view_claim/<int:claim_id>')
def view_claim(claim_id):
    claim = Claim.query.get_or_404(claim_id)

    # Retrieve recommendation from session
    recommendation = session.get(f"recommendation_{claim_id}", None)

    # Render the view.html with claim details and the recommendation (if available)
    return render_template('view.html', 
                           claim=claim,
                           recommendation=recommendation)


@app.route('/agent_decision', methods=['POST'])
def agent_decision():
    claim_id = request.form.get('claim_id')
    decision = request.form.get('decision')
    notes = request.form.get('notes')

    # Fetch the claim from the database
    claim = Claim.query.get_or_404(claim_id)

    # Update claim with the agent's decision
    claim.status = decision
    claim.notes = notes

    try:
        db.session.commit()
        flash('Decision recorded successfully')
    except Exception as e:
        db.session.rollback()
        flash('Error recording decision')
        print(e)

    # Redirect back to the agent dashboard
    return redirect(url_for('agent_dashboard'))


# Run the app
if __name__ == '__main__':
    
    app.run(debug=True)

