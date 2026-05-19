
"""
AI Doctor - Smart Pain Diagnosis System
Main Application File
"""

import sys
import os
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import pickle
    from datetime import datetime
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Please install requirements:")
    print("pip install -r requirements.txt")
    sys.exit(1)

# -------------------- Configuration --------------------
# Set page config early
st.set_page_config(
    page_title="🤖 AI Doctor - Smart Diagnosis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Custom styling */
    .main-title {
        text-align: center;
        font-size: 3rem;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #4CAF50;
    }
    
    .warning-card {
        background: #FFF3CD;
        border-left: 5px solid #FFC107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .info-card {
        background: #D1ECF1;
        border-left: 5px solid #17A2B8;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .success-card {
        background: #D4EDDA;
        border-left: 5px solid #28A745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea, #764ba2);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Table styling */
    .dataframe {
        width: 100%;
    }
    
    /* Custom metrics */
    .metric-card {
        text-align: center;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# -------------------- File Loading Functions --------------------
@st.cache_resource
def load_model():
    """Load the trained ML model"""
    try:
        model_path = project_dir / "model" / "doctor_model.pkl"
        if not model_path.exists():
            st.error("❌ Model file not found. Please run train_model.py first.")
            return None
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_features():
    """Load feature names for symptoms"""
    try:
        features_path = project_dir / "model" / "feature_names.pkl"
        if not features_path.exists():
            st.error("❌ Feature names file not found.")
            return []
        
        with open(features_path, "rb") as f:
            features = pickle.load(f)
        
        st.success(f"✅ Loaded {len(features)} symptoms")
        return features
    except Exception as e:
        st.error(f"❌ Error loading features: {str(e)}")
        return []

@st.cache_data
def load_medical_data():
    """Load medical database"""
    try:
        data_path = project_dir / "cleaned_medical_data_with_food.csv"
        if not data_path.exists():
            st.error("❌ Medical data CSV not found.")
            return pd.DataFrame()
        
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.strip().str.lower()
        
        # Clean data
        df['disease'] = df['disease'].astype(str).str.strip().str.lower()
        df['symptoms'] = df['symptoms'].fillna('').astype(str)
        
        st.success(f"✅ Loaded {len(df)} medical records")
        return df
    except Exception as e:
        st.error(f"❌ Error loading medical data: {str(e)}")
        return pd.DataFrame()

# -------------------- Helper Functions --------------------
def save_patient_info(name, age, gender, city, state, country):
    """Save patient information"""
    try:
        # Create data directory
        data_dir = project_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        file_path = data_dir / "patient_records.xlsx"
        
        # Create new record
        new_record = pd.DataFrame([{
            "Name": name,
            "Age": int(age),
            "Gender": gender,
            "City": city,
            "State": state,
            "Country": country,
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
        
        # Load existing records or create new
        if file_path.exists():
            existing = pd.read_excel(file_path)
            updated = pd.concat([existing, new_record], ignore_index=True)
        else:
            updated = new_record
        
        # Save
        updated.to_excel(file_path, index=False)
        return True, "Patient information saved successfully!"
    except Exception as e:
        return False, f"Error: {str(e)}"

def get_patient_records():
    """Load patient records"""
    try:
        file_path = project_dir / "data" / "patient_records.xlsx"
        if file_path.exists():
            return pd.read_excel(file_path)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading records: {e}")
        return pd.DataFrame()

def get_disease_info(medical_df, disease_name):
    """Get information about a specific disease"""
    if medical_df.empty:
        return {}
    
    disease_lower = disease_name.strip().lower()
    matches = medical_df[medical_df['disease'].str.contains(disease_lower, case=False, na=False)]
    
    if not matches.empty:
        row = matches.iloc[0]
        return {
            'causes': row.get('causes', 'Not available'),
            'medicine': row.get('medicine', 'Not available'),
            'food_intake': row.get('food_intake', 'Not available'),
            'food_avoid': row.get('food_avoid', 'Not available')
        }
    return {}

# -------------------- Initialize Application --------------------
def initialize_app():
    """Initialize the application"""
    # Check for required files
    required_files = [
        project_dir / "cleaned_medical_data_with_food.csv",
        project_dir / "model" / "doctor_model.pkl",
        project_dir / "model" / "feature_names.pkl"
    ]
    
    missing_files = [str(f.name) for f in required_files if not f.exists()]
    
    if missing_files:
        st.error(f"❌ Missing required files: {', '.join(missing_files)}")
        st.info("Please ensure:")
        st.info("1. CSV file is in the project folder")
        st.info("2. Run train_model.py to create model files")
        return None, None, None
    
    # Load data
    with st.spinner("Loading AI Doctor system..."):
        model = load_model()
        features = load_features()
        medical_data = load_medical_data()
    
    return model, features, medical_data

# -------------------- Page Functions --------------------
def home_page():
    """Home page content"""
    st.markdown('<h1 class="main-title">🤖 AI Doctor</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #666;">Smart Pain Diagnosis System</h3>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Available Symptoms", len(st.session_state.features))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        disease_count = len(st.session_state.medical_data['disease'].unique()) if not st.session_state.medical_data.empty else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Diseases in Database", disease_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        records = get_patient_records()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Patients", len(records))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("AI Model", "Ready ✅")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # How it works
    st.subheader("🎯 How It Works")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4>👤 1. Patient Info</h4>
            <p>Register new patients with their details</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4>🩺 2. Symptom Analysis</h4>
            <p>Select symptoms from comprehensive list</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <h4>💊 3. AI Diagnosis</h4>
            <p>Get AI-powered diagnosis with recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start
    st.markdown("---")
    st.subheader("⚡ Quick Start Guide")
    
    with st.expander("Click to expand"):
        st.markdown("""
        1. **Go to Patient Information** page
        2. **Register** a new patient
        3. **Navigate** to Diagnosis page
        4. **Select symptoms** from the list
        5. **Click** Get Diagnosis button
        6. **Review** AI-powered results
        
        **Note:** This system is for educational purposes only.
        Always consult healthcare professionals for medical advice.
        """)

def patient_info_page():
    """Patient information page"""
    st.header("👤 Patient Registration")
    
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name *", placeholder="John Doe")
            age = st.number_input("Age *", min_value=0, max_value=120, value=25)
            gender = st.selectbox("Gender *", ["Male", "Female", "Other", "Prefer not to say"])
        
        with col2:
            city = st.text_input("City", placeholder="New York")
            state = st.text_input("State/Province", placeholder="NY")
            country = st.text_input("Country", placeholder="USA")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("💾 Save Patient Information", use_container_width=True)
        
        if submitted:
            if name and age:
                success, message = save_patient_info(name, age, gender, city, state, country)
                if success:
                    st.success(f"✅ {message}")
                    st.session_state.patient_name = name
                    st.info(f"Proceed to Diagnosis tab for **{name}**")
                else:
                    st.error(f"❌ {message}")
            else:
                st.warning("⚠️ Please fill in at least Name and Age fields")
    
    # Show recent patients
    st.markdown("---")
    st.subheader("📋 Recent Patients")
    
    records = get_patient_records()
    if not records.empty:
        # Show last 5 records
        recent = records.tail(5)
        st.dataframe(
            recent,
            use_container_width=True,
            hide_index=True
        )
        st.caption(f"Showing 5 of {len(records)} total patients")
    else:
        st.info("No patients registered yet. Fill the form above to add patients.")

def diagnosis_page():
    """Diagnosis page"""
    st.header("🩺 Smart Diagnosis")
    
    # Sidebar content
    with st.sidebar:
        st.markdown("### 📍 Pain Details")
        
        pain_area = st.selectbox("Location", [
            "Head", "Chest", "Abdomen", "Back", "Leg", "Arm", 
            "Neck", "Shoulder", "Eye", "Ear", "Throat", "Multiple"
        ])
        
        pain_level = st.slider("Pain Level (0-10)", 0, 10, 3)
        
        pain_duration = st.selectbox("Duration", [
            "Less than 1 hour", "1-6 hours", "6-12 hours", 
            "12-24 hours", "1-3 days", "More than 3 days"
        ])
        
        st.markdown("---")
        st.markdown("### 👤 Patient")
        
        # Patient selection
        records = get_patient_records()
        if not records.empty:
            patient_names = ["Select Patient"] + records["Name"].unique().tolist()
            selected_patient = st.selectbox("Choose Patient", patient_names)
            
            if selected_patient != "Select Patient":
                st.session_state.patient_name = selected_patient
                patient_info = records[records["Name"] == selected_patient].iloc[-1]
                st.info(f"**Selected:** {selected_patient}")
        else:
            st.info("No patients found. Add patients first.")
    
    # Main content - Symptoms selection
    st.markdown("### 🧠 Select Your Symptoms")
    
    # Organize symptoms into groups
    num_groups = min(5, (len(st.session_state.features) // 25) + 1)
    group_size = len(st.session_state.features) // num_groups
    
    selected_symptoms = []
    
    # Create tabs for symptom groups
    tabs = st.tabs([f"Group {i+1}" for i in range(num_groups)])
    
    for i, tab in enumerate(tabs):
        with tab:
            start_idx = i * group_size
            end_idx = start_idx + group_size if i < num_groups - 1 else len(st.session_state.features)
            symptoms = st.session_state.features[start_idx:end_idx]
            
            selected = st.multiselect(
                f"Select symptoms (Group {i+1})",
                symptoms,
                key=f"group_{i}"
            )
            selected_symptoms.extend(selected)
    
    # Additional symptoms
    with st.expander("➕ Add Other Symptoms"):
        other_symptoms = st.text_area(
            "Enter symptoms not in the list (one per line):",
            height=100,
            placeholder="Example:\nloss of appetite\nnight sweats\nmuscle weakness"
        )
        if other_symptoms:
            additional = [s.strip() for s in other_symptoms.split("\n") if s.strip()]
            selected_symptoms.extend(additional)
    
    # Show selected symptoms
    if selected_symptoms:
        st.markdown(f"### ✅ Selected Symptoms ({len(selected_symptoms)})")
        cols = st.columns(3)
        for idx, symptom in enumerate(selected_symptoms):
            with cols[idx % 3]:
                st.markdown(f"• {symptom}")
    
    # Diagnosis button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        diagnose_btn = st.button(
            "🔍 GET DIAGNOSIS",
            type="primary",
            use_container_width=True,
            disabled=len(selected_symptoms) == 0
        )
    
    # Perform diagnosis
    if diagnose_btn and selected_symptoms:
        st.markdown("---")
        
        with st.spinner("🤖 AI is analyzing symptoms..."):
            try:
                # Create input vector
                input_vector = {symptom: 1 if symptom in selected_symptoms else 0 
                              for symptom in st.session_state.features}
                input_df = pd.DataFrame([input_vector])
                
                # Predict
                prediction = st.session_state.model.predict(input_df)[0]
                
                # Display results
                st.markdown("""
                <div style="background: linear-gradient(90deg, #667eea, #764ba2);
                           color: white; padding: 2rem; border-radius: 10px; text-align: center;">
                    <h2>🧬 DIAGNOSIS RESULT</h2>
                    <h1 style="margin: 0;">{}</h1>
                </div>
                """.format(prediction.title()), unsafe_allow_html=True)
                
                # Get disease information
                disease_info = get_disease_info(st.session_state.medical_data, prediction)
                
                # Display in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📋 Patient Details")
                    if "patient_name" in st.session_state and st.session_state.patient_name:
                        st.markdown(f"**Patient:** {st.session_state.patient_name}")
                    
                    st.markdown(f"**Symptoms Selected:** {len(selected_symptoms)}")
                    st.markdown(f"**Pain Location:** {pain_area}")
                    st.markdown(f"**Pain Level:** {pain_level}/10")
                    
                    if pain_level >= 7:
                        st.error("⚠️ **Severe Pain Alert** - Consider seeking medical attention")
                
                with col2:
                    st.markdown("### 💡 Recommendations")
                    
                    if disease_info:
                        if disease_info['medicine'] != 'Not available':
                            st.markdown(f"**Medication:** {disease_info['medicine']}")
                        
                        if disease_info['causes'] != 'Not available':
                            st.markdown(f"**Possible Causes:** {disease_info['causes']}")
                        
                        if disease_info['food_intake'] != 'Not available':
                            st.markdown(f"**Foods to Eat:** {disease_info['food_intake']}")
                        
                        if disease_info['food_avoid'] != 'Not available':
                            st.markdown(f"**Foods to Avoid:** {disease_info['food_avoid']}")
                    else:
                        st.markdown("""
                        **General Advice:**
                        - Rest and stay hydrated
                        - Monitor symptoms
                        - Consult healthcare provider
                        - Avoid self-medication
                        """)
                
            except Exception as e:
                st.error(f"❌ Error during diagnosis: {str(e)}")
                st.info("Please try selecting different symptoms or consult a doctor.")

def records_page():
    """Patient records page"""
    st.header("📊 Patient Records")
    
    records = get_patient_records()
    
    if not records.empty:
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", len(records))
        
        with col2:
            avg_age = records["Age"].mean()
            st.metric("Average Age", f"{avg_age:.1f}")
        
        with col3:
            male_count = len(records[records["Gender"] == "Male"])
            st.metric("Male Patients", male_count)
        
        with col4:
            female_count = len(records[records["Gender"] == "Female"])
            st.metric("Female Patients", female_count)
        
        # Data table
        st.dataframe(
            records,
            use_container_width=True,
            hide_index=True
        )
        
        # Export options
        st.markdown("---")
        st.subheader("📥 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = records.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="patient_records.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Create Excel file in memory
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                records.to_excel(writer, index=False, sheet_name='Patients')
            excel_data = output.getvalue()
            
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name="patient_records.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    else:
        st.info("📭 No patient records found.")
        st.markdown("""
        To add patient records:
        1. Go to **Patient Information** page
        2. Fill in the patient details
        3. Click **Save Patient Information**
        """)

# -------------------- Main Application --------------------
def main():
    """Main application function"""
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'patient_name' not in st.session_state:
        st.session_state.patient_name = ""
    
    # Load data if not loaded
    if not st.session_state.initialized:
        with st.spinner("🚀 Initializing AI Doctor System..."):
            model, features, medical_data = initialize_app()
            
            if model is None or features is None or medical_data is None:
                st.error("Failed to initialize application. Please check the console for errors.")
                return
            
            st.session_state.model = model
            st.session_state.features = features
            st.session_state.medical_data = medical_data
            st.session_state.initialized = True
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        
        page = st.radio(
            "Go to:",
            ["🏠 Home", "👤 Patient Info", "🩺 Diagnosis", "📊 Records"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # System info
        st.markdown("### 📊 System Info")
        st.markdown(f"**Symptoms:** {len(st.session_state.features)}")
        st.markdown(f"**Diseases:** {len(st.session_state.medical_data['disease'].unique())}")
        
        if st.session_state.patient_name:
            st.markdown(f"**Current Patient:** {st.session_state.patient_name}")
    
    # Display selected page
    if page == "🏠 Home":
        home_page()
    elif page == "👤 Patient Info":
        patient_info_page()
    elif page == "🩺 Diagnosis":
        diagnosis_page()
    elif page == "📊 Records":
        records_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🤖 <strong>AI Doctor System</strong> • Version 1.0 • Educational Use Only</p>
        <p><small>⚠️ Always consult healthcare professionals for medical advice</small></p>
    </div>
    """, unsafe_allow_html=True)

# -------------------- Entry Point --------------------
if __name__ == "__main__":
    # Set working directory
    os.chdir(project_dir)
    
    # Run the main app
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("""
        If you see import errors:
        1. Open terminal in VS Code (Ctrl+`)
        2. Run: `pip install -r requirements.txt`
        3. Restart the app: `streamlit run app.py`
        """)

# -------------------- VS Code Specific Fix --------------------
# This helps with Pylance intellisense
if False:  # This block is only for type hints
    import streamlit as st
    import pandas as pd
    import numpy as np
    import pickle