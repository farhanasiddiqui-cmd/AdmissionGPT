import os
import json
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from datetime import datetime
import sqlite3

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
CORS(app)

# ============================================================================
# AGENT_INSTRUCTIONS - Customize AI Admission Advisor Behavior
# ============================================================================
AGENT_INSTRUCTIONS = """
You are an expert B.E. (Bachelor of Engineering) Admission Guidance Counselor specializing in Mumbai engineering colleges and Maharashtra CAP (Centralized Admission Process) rounds.

## Your Role:
- Provide personalized college recommendations based on student marks, category, preferences
- Guide students through CAP rounds (CAP Round 1, 2, 3, and Institutional rounds)
- Suggest suitable engineering branches based on interests and career goals
- Analyze cut-off trends and predict admission chances
- Provide document checklists and fee estimates
- Offer admission planning and timeline guidance

## Counseling Style:
- Be friendly, encouraging, and supportive
- Use simple language that students and parents can understand
- Provide data-driven recommendations with reasoning
- Be honest about admission chances (high/medium/low probability)
- Consider both academic performance and student preferences

## Recommendation Criteria:
1. **Academic Performance**: MHT-CET/JEE Main percentile, 12th marks
2. **Category**: OPEN, OBC, SC, ST, EWS, TFWS
3. **Location Preference**: Mumbai, Navi Mumbai, Thane, Pune, Other Maharashtra
4. **Branch Interest**: Computer, IT, Electronics, Mechanical, Civil, etc.
5. **College Type**: Government, Autonomous, Private Aided, Private Unaided
6. **Budget**: Fee range consideration
7. **Placement Records**: Average package, top recruiters
8. **Infrastructure**: Labs, library, hostel facilities

## Mumbai Engineering Colleges Priority (Top Tier):
- VJTI (Veermata Jijabai Technological Institute)
- SPIT (Sardar Patel Institute of Technology)
- DJ Sanghvi (Dwarkadas J. Sanghvi College of Engineering)
- KJ Somaiya (K.J. Somaiya College of Engineering)
- TSEC (Thadomal Shahani Engineering College)
- Terna Engineering College
- Fr. Conceicao Rodrigues College of Engineering
- Atharva College of Engineering

## Safety Rules:
- Never guarantee admission - always mention it depends on actual cut-offs
- Don't provide false hope - be realistic about chances
- Recommend backup options and multiple colleges
- Suggest applying to mix of safe, moderate, and ambitious colleges
- Remind about document verification and important deadlines
- Advise consulting official DTE Maharashtra website for final information
- Don't ask for sensitive personal information beyond academic details

## Mumbai Admission Preferences:
- Prioritize colleges with good connectivity (near local trains/metro)
- Consider hostel availability for outstation students
- Mention college reputation and alumni network
- Highlight industry connections and internship opportunities
- Consider college culture and extracurricular activities
- Provide realistic fee estimates including hidden costs

## Response Format:
When providing recommendations, structure your response with:
1. Greeting and acknowledgment of student's profile
2. Admission probability assessment
3. Top 5-7 college recommendations with reasons
4. Branch suggestions based on interests
5. CAP round strategy
6. Document checklist
7. Fee estimation
8. Next steps and timeline

Always be encouraging while being realistic. Your goal is to help students make informed decisions.
"""

# ============================================================================
# IBM Watsonx.ai Configuration
# ============================================================================

def get_watsonx_model():
    """Initialize and return IBM Watsonx.ai model"""
    api_key = os.getenv('IBM_CLOUD_API_KEY')
    project_id = os.getenv('WATSONX_PROJECT_ID')
    
    if not api_key or not project_id:
        raise ValueError("IBM_CLOUD_API_KEY and WATSONX_PROJECT_ID must be set in .env file")
    
    model_id = "ibm/granite-13b-chat-v2"
    
    parameters = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MAX_NEW_TOKENS: 2000,
        GenParams.MIN_NEW_TOKENS: 50,
        GenParams.TEMPERATURE: 0.7,
        GenParams.TOP_K: 50,
        GenParams.TOP_P: 1
    }
    
    model = Model(
        model_id=model_id,
        params=parameters,
        credentials={
            "apikey": api_key,
            "url": "https://us-south.ml.cloud.ibm.com"
        },
        project_id=project_id
    )
    
    return model

# ============================================================================
# Database Functions
# ============================================================================

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect('admission_guidance.db')
    c = conn.cursor()
    
    # Students table
    c.execute('''CREATE TABLE IF NOT EXISTS students
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  email TEXT UNIQUE,
                  phone TEXT,
                  mht_cet_percentile REAL,
                  jee_main_percentile REAL,
                  hsc_percentage REAL,
                  category TEXT,
                  location_preference TEXT,
                  branch_interest TEXT,
                  budget_range TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Chat history table
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  student_id INTEGER,
                  message TEXT,
                  response TEXT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (student_id) REFERENCES students(id))''')
    
    # College recommendations table
    c.execute('''CREATE TABLE IF NOT EXISTS recommendations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  student_id INTEGER,
                  college_name TEXT,
                  branch TEXT,
                  probability TEXT,
                  cutoff_estimate TEXT,
                  fees_estimate TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (student_id) REFERENCES students(id))''')
    
    conn.commit()
    conn.close()

def save_student_profile(data):
    """Save student profile to database"""
    conn = sqlite3.connect('admission_guidance.db')
    c = conn.cursor()
    
    try:
        c.execute('''INSERT INTO students 
                     (name, email, phone, mht_cet_percentile, jee_main_percentile, 
                      hsc_percentage, category, location_preference, branch_interest, budget_range)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (data.get('name'), data.get('email'), data.get('phone'),
                   data.get('mht_cet_percentile'), data.get('jee_main_percentile'),
                   data.get('hsc_percentage'), data.get('category'),
                   data.get('location_preference'), data.get('branch_interest'),
                   data.get('budget_range')))
        conn.commit()
        student_id = c.lastrowid
        return student_id
    except sqlite3.IntegrityError:
        # Email already exists, get existing student
        c.execute('SELECT id FROM students WHERE email = ?', (data.get('email'),))
        result = c.fetchone()
        return result[0] if result else None
    finally:
        conn.close()

def save_chat_history(student_id, message, response):
    """Save chat interaction to database"""
    conn = sqlite3.connect('admission_guidance.db')
    c = conn.cursor()
    c.execute('INSERT INTO chat_history (student_id, message, response) VALUES (?, ?, ?)',
              (student_id, message, response))
    conn.commit()
    conn.close()

# ============================================================================
# Routes
# ============================================================================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Admission dashboard"""
    return render_template('dashboard.html')

@app.route('/chatbot')
def chatbot():
    """Chatbot interface"""
    return render_template('chatbot.html')

@app.route('/predictor')
def predictor():
    """College predictor"""
    return render_template('predictor.html')

@app.route('/eligibility')
def eligibility():
    """Eligibility checker"""
    return render_template('eligibility.html')

@app.route('/cutoff')
def cutoff():
    """Cut-off analysis"""
    return render_template('cutoff.html')

@app.route('/profile')
def profile():
    """Student profile"""
    return render_template('profile.html')

# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages with AI"""
    try:
        data = request.json
        user_message = data.get('message', '')
        student_profile = data.get('profile', {})
        
        # Build context with student profile
        context = f"{AGENT_INSTRUCTIONS}\n\n"
        if student_profile:
            context += "Student Profile:\n"
            for key, value in student_profile.items():
                context += f"- {key}: {value}\n"
            context += "\n"
        
        context += f"Student Query: {user_message}\n\nProvide helpful admission guidance:"
        
        # Get AI response
        model = get_watsonx_model()
        response = model.generate_text(prompt=context)
        
        # Save to database if student_id exists
        if 'student_id' in data:
            save_chat_history(data['student_id'], user_message, response)
        
        return jsonify({
            'success': True,
            'response': response
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict_colleges():
    """Predict colleges based on student data"""
    try:
        data = request.json
        
        # Build prediction prompt
        prompt = f"""{AGENT_INSTRUCTIONS}

Student Details:
- MHT-CET Percentile: {data.get('mht_cet_percentile', 'N/A')}
- JEE Main Percentile: {data.get('jee_main_percentile', 'N/A')}
- 12th Percentage: {data.get('hsc_percentage', 'N/A')}
- Category: {data.get('category', 'OPEN')}
- Location Preference: {data.get('location_preference', 'Mumbai')}
- Branch Interest: {data.get('branch_interest', 'Computer Engineering')}
- Budget Range: {data.get('budget_range', 'Any')}

Provide a detailed college prediction with:
1. Top 7 recommended colleges in Mumbai/Maharashtra
2. Admission probability for each (High/Medium/Low)
3. Expected cut-off range
4. Estimated fees
5. Branch recommendations
6. CAP round strategy

Format as a structured list."""

        model = get_watsonx_model()
        response = model.generate_text(prompt=prompt)
        
        return jsonify({
            'success': True,
            'predictions': response
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/check-eligibility', methods=['POST'])
def check_eligibility():
    """Check eligibility for engineering admission"""
    try:
        data = request.json
        
        prompt = f"""{AGENT_INSTRUCTIONS}

Check eligibility for B.E. admission:
- 12th Percentage: {data.get('hsc_percentage')}
- Physics Marks: {data.get('physics_marks')}
- Chemistry Marks: {data.get('chemistry_marks')}
- Mathematics Marks: {data.get('mathematics_marks')}
- Category: {data.get('category')}
- Domicile: {data.get('domicile', 'Maharashtra')}

Provide:
1. Eligibility status (Eligible/Not Eligible)
2. Minimum requirements check
3. Exam options (MHT-CET/JEE Main)
4. Documents needed
5. Next steps"""

        model = get_watsonx_model()
        response = model.generate_text(prompt=prompt)
        
        return jsonify({
            'success': True,
            'eligibility': response
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analyze-cutoff', methods=['POST'])
def analyze_cutoff():
    """Analyze cut-off trends"""
    try:
        data = request.json
        
        prompt = f"""{AGENT_INSTRUCTIONS}

Analyze cut-off trends for:
- College: {data.get('college_name', 'Mumbai Engineering Colleges')}
- Branch: {data.get('branch', 'Computer Engineering')}
- Category: {data.get('category', 'OPEN')}
- Year Range: Last 3 years

Provide:
1. Historical cut-off trends
2. Expected cut-off for current year
3. Factors affecting cut-offs
4. Comparison with similar colleges
5. Admission strategy"""

        model = get_watsonx_model()
        response = model.generate_text(prompt=prompt)
        
        return jsonify({
            'success': True,
            'analysis': response
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/save-profile', methods=['POST'])
def save_profile():
    """Save student profile"""
    try:
        data = request.json
        student_id = save_student_profile(data)
        
        return jsonify({
            'success': True,
            'student_id': student_id,
            'message': 'Profile saved successfully'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/get-documents', methods=['GET'])
def get_documents():
    """Get document checklist"""
    prompt = f"""{AGENT_INSTRUCTIONS}

Provide a comprehensive document checklist for B.E. admission in Maharashtra through CAP process.
Include:
1. Essential documents
2. Optional documents
3. Document format requirements
4. Verification process
5. Common mistakes to avoid"""

    try:
        model = get_watsonx_model()
        response = model.generate_text(prompt=prompt)
        
        return jsonify({
            'success': True,
            'documents': response
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/estimate-fees', methods=['POST'])
def estimate_fees():
    """Estimate college fees"""
    try:
        data = request.json
        
        prompt = f"""{AGENT_INSTRUCTIONS}

Estimate fees for:
- College Type: {data.get('college_type', 'Private Unaided')}
- Branch: {data.get('branch', 'Computer Engineering')}
- Location: {data.get('location', 'Mumbai')}
- Category: {data.get('category', 'OPEN')}

Provide:
1. Tuition fees (per year)
2. Development fees
3. Exam fees
4. Hostel fees (if applicable)
5. Other expenses
6. Total estimated cost for 4 years
7. Scholarship opportunities"""

        model = get_watsonx_model()
        response = model.generate_text(prompt=prompt)
        
        return jsonify({
            'success': True,
            'fee_estimate': response
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================================================
# Initialize and Run
# ============================================================================

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)

# Made with Bob
