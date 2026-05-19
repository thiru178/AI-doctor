import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import os

# Set your project path
project_path = r"C:\Users\Asus\Documents\ai doctor"

# Load and prepare data
df = pd.read_csv(os.path.join(project_path, "cleaned_medical_data_with_food.csv"))
df.columns = df.columns.str.strip().str.lower()

# Clean disease column
df['disease'] = df['disease'].astype(str).str.strip().str.lower()

# Print basic info
print(f"Total records in dataset: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Create features (symptoms)
all_symptoms = []
for symptoms in df['symptoms'].dropna():
    # Split by comma and clean
    symptom_list = str(symptoms).split(',')
    for symptom in symptom_list:
        clean_symptom = symptom.strip().lower()
        if clean_symptom and len(clean_symptom) > 2:  # Changed to >2 to filter out short strings
            all_symptoms.append(clean_symptom)

# Get unique symptoms
unique_symptoms = list(set(all_symptoms))
feature_names = sorted([s for s in unique_symptoms if s and len(s) > 2])  # Sort for consistency

print(f"\nTotal unique symptoms found: {len(feature_names)}")
print(f"First 10 symptoms: {feature_names[:10]}")

# Create binary features for each symptom
feature_data = []
diseases = []
skipped_rows = 0

for idx, row in df.iterrows():
    if pd.isna(row['symptoms']) or pd.isna(row['disease']):
        skipped_rows += 1
        continue
    
    disease = str(row['disease']).strip().lower()
    symptoms_list = [s.strip().lower() for s in str(row['symptoms']).split(',')]
    
    # Create feature vector
    features = [1 if symptom in symptoms_list else 0 for symptom in feature_names]
    
    feature_data.append(features)
    diseases.append(disease)

print(f"\nCreated {len(feature_data)} training samples")
print(f"Skipped {skipped_rows} rows due to missing data")

# Check if we have enough data
if len(feature_data) == 0:
    print("❌ ERROR: No valid training data found!")
    print("Possible issues:")
    print("1. CSV file might be empty or corrupted")
    print("2. 'symptoms' column might have no valid data")
    print("3. 'disease' column might have no valid data")
    exit()

# Train model
print("\n🤖 Training model...")
model = DecisionTreeClassifier(max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42)
model.fit(feature_data, diseases)

# Model info
print(f"Model trained successfully!")
print(f"Number of features: {len(feature_names)}")
print(f"Number of unique diseases: {len(set(diseases))}")

# Create model directory if it doesn't exist
model_dir = os.path.join(project_path, "model")
os.makedirs(model_dir, exist_ok=True)

# Save model with correct name
model_path = os.path.join(model_dir, "doctor_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"✅ Model saved to: {model_path}")

# Save feature names
features_path = os.path.join(model_dir, "feature_names.pkl")
with open(features_path, "wb") as f:
    pickle.dump(feature_names, f)
print(f"✅ Feature names saved to: {features_path}")

# Create a simple test to verify the model
print("\n🧪 Testing the model...")

# Test with some sample symptoms
if len(feature_data) > 0:
    # Test with first few training samples
    test_sample = feature_data[0]
    prediction = model.predict([test_sample])[0]
    actual = diseases[0]
    
    print(f"Test Prediction: {prediction}")
    print(f"Actual Disease: {actual}")
    
    # Test accuracy on training data (for verification only)
    train_accuracy = model.score(feature_data, diseases)
    print(f"Training accuracy: {train_accuracy:.2%}")

# Print summary
print("\n" + "="*50)
print("TRAINING SUMMARY:")
print("="*50)
print(f"• Dataset: {len(df)} records")
print(f"• Valid training samples: {len(feature_data)}")
print(f"• Number of symptoms (features): {len(feature_names)}")
print(f"• Number of diseases: {len(set(diseases))}")
print(f"• Common diseases: {sorted(set(diseases))[:5]}")
print(f"• Model saved to: {model_path}")
print(f"• Features saved to: {features_path}")
print("="*50)
print("✅ Training completed successfully!")
