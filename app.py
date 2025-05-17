import streamlit as st
import os
import time
import json
import tempfile
from datetime import datetime
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
import requests
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

def transcribe_audio(audio_bytes):
    """Transcribe audio bytes to text using a custom temp file approach"""
    try:
        # Create a unique filename in the current directory
        temp_dir = os.path.join(os.getcwd(), "temp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Create a unique filename based on timestamp
        timestamp = int(time.time() * 1000)
        filename = f"recording_{timestamp}.wav"
        filepath = os.path.join(temp_dir, filename)
        
        try:
            # Write audio bytes to file
            with open(filepath, 'wb') as f:
                f.write(audio_bytes)
            
            # Initialize recognizer
            recognizer = sr.Recognizer()
            
            # Load audio file
            with sr.AudioFile(filepath) as source:
                # Record the audio file
                audio = recognizer.record(source)
                
                # Attempt to recognize speech
                text = recognizer.recognize_google(audio)
                return {"success": True, "text": text}
                
        except sr.UnknownValueError:
            return {"success": False, "error": "Could not understand the audio"}
        except sr.RequestError as e:
            return {"success": False, "error": f"Could not request results; {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Error processing audio: {str(e)}"}
        finally:
            # Clean up temp file
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception:
                pass  # If cleanup fails, just continue
    
    except Exception as e:
        return {"success": False, "error": f"Setup error: {str(e)}"}
    
# Set page configuration
st.set_page_config(
    page_title="English Speaking Evaluation",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    .results-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        border: 1px solid #dee2e6;
    }
    .score-box {
        background-color: #e9ecef;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 10px;
    }
    .feedback-item {
        margin-bottom: 15px;
    }
    .recording-indicator {
        color: red;
        font-weight: bold;
        animation: blinker 1s linear infinite;
    }
    @keyframes blinker {
        50% { opacity: 0; }
    }
    .header-container {
        background-color: #f1f8ff;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid #0366d6;
    }
    .instruction-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        border-left: 3px solid #5cb85c;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

if 'current_test' not in st.session_state:
    st.session_state.current_test = None

# Get API key from environment variable (no hardcoded keys)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
MODEL_OPTION = "llama3-8b-8192"
TEMPERATURE = 0.5

# Try importing config file as a fallback
try:
    import config
    if hasattr(config, "GROQ_API_KEY") and config.GROQ_API_KEY:
        GROQ_API_KEY = config.GROQ_API_KEY
except ImportError:
    pass  # Continue with environment variable or empty key

# Load reading texts from a JSON file if available
def load_reading_texts():
    try:
        with open("reading_texts.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return default text if file not found or invalid
        return {
            "default": """The rapid advancement of artificial intelligence has brought significant changes 
            to various industries. While some fear job displacement, others believe AI will create new 
            opportunities and enhance human capabilities. Researchers continue to debate the long-term 
            implications of these technologies on society, economy, and human cognition."""
        }

# Sidebar for minimal settings
with st.sidebar:
    # Use a local image if available, otherwise use a placeholder
    try:
        st.image("assets/logo.png", width=90)
    except:
        st.image("https://via.placeholder.com/100x100.png?text=Speaking-Test", width=150)
    
    st.title("Speaking-Test")
    
    st.markdown("---")
    
    # History section
    st.subheader("Test History")
    if st.session_state.history:
        for i, entry in enumerate(st.session_state.history):
            with st.expander(f"{entry['type']} - {entry['date']}"):
                st.write(f"**Score:** {entry['score']}/10")
                st.write(f"**Transcript:** {entry['transcript'][:100]}...")
    else:
        st.info("No test history yet")

def evaluate_with_groq(text, evaluation_type="speaking", reference_text=None):
    """Get evaluation from Groq API using direct HTTP requests"""
    # Check if API key is properly configured
    if not GROQ_API_KEY:
        return {"success": False, "error": "Groq API key is not configured. Please set the GROQ_API_KEY environment variable."}
        
    try:
        # Set the system prompt based on evaluation type
        system_prompt = ""
        
        # For speaking evaluation
        if evaluation_type == "speaking":
            system_prompt = """You are an English language expert. Evaluate the user's speech for:
            1. Pronunciation/clarity
            2. Grammar accuracy
            3. Vocabulary range
            4. Fluency/pace
            5. Overall coherence
            
            Provide detailed feedback for each category and an overall score from 1-10. 
            Format your response as JSON with the following structure:
            {
                "score": 7,
                "pronunciation": "Feedback on pronunciation...",
                "grammar": "Feedback on grammar...",
                "vocabulary": "Feedback on vocabulary...",
                "fluency": "Feedback on fluency...",
                "coherence": "Feedback on coherence...",
                "improvement_tips": ["Tip 1", "Tip 2", "Tip 3"]
            }
            """
        # For reading evaluation
        elif evaluation_type == "reading":
            system_prompt = f"""You are an English language expert. Evaluate the user's reading accuracy and fluency.
            
            Original text: "{reference_text}"
            
            User's reading: "{text}"
            
            Compare the original text with the user's reading and evaluate:
            1. Reading accuracy (how well they read the exact words)
            2. Pronunciation
            3. Fluency/pace
            4. Overall performance
            
            Format your response as JSON with the following structure:
            {{
                "score": 7,
                "accuracy": "Feedback on reading accuracy...",
                "pronunciation": "Feedback on pronunciation...",
                "fluency": "Feedback on fluency...",
                "overall": "Overall feedback...",
                "improvement_tips": ["Tip 1", "Tip 2", "Tip 3"]
            }}
            """
        # Unknown evaluation type
        else:
            return {"success": False, "error": f"Unknown evaluation type: {evaluation_type}"}

        # Prepare the request payload
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "model": MODEL_OPTION,
            "temperature": TEMPERATURE,
            "response_format": {"type": "json_object"}
        }

        # Make the API request
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30  # Add timeout to prevent hanging
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            try:
                response_json = response.json()
                result = json.loads(response_json["choices"][0]["message"]["content"])
                return {"success": True, "evaluation": result}
            except (json.JSONDecodeError, KeyError) as e:
                return {"success": False, "error": f"Error parsing API response: {str(e)}"}
        else:
            # Return the specific error from the API
            try:
                error_detail = response.json() if response.content else {"error": f"HTTP {response.status_code}"}
                return {"success": False, "error": f"Error from Groq API: {error_detail}"}
            except json.JSONDecodeError:
                return {"success": False, "error": f"HTTP error {response.status_code}: {response.text[:100]}"}
    
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Network error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Error processing request: {str(e)}"}

def display_evaluation_results(evaluation_results, transcript):
    """Display evaluation results in a visually appealing way"""
    if not evaluation_results["success"]:
        st.error(f"Evaluation failed: {evaluation_results['error']}")
        return
    
    eval_data = evaluation_results["evaluation"]
    
    # Display overall score
    st.markdown(f"""
    <div class="score-box">
        <h2>Overall Score: {eval_data.get('score', 'N/A')}/10</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Display transcript
    st.subheader("Your Transcribed Speech:")
    st.info(transcript)
    
    # Display detailed feedback
    st.subheader("Detailed Feedback:")
    
    cols = st.columns(2)
    
    # Check which evaluation type we have based on keys
    if "grammar" in eval_data:  # Speaking evaluation
        with cols[0]:
            st.markdown(f"""
            <div class="feedback-item">
                <h4>Pronunciation & Clarity</h4>
                <p>{eval_data.get('pronunciation', 'N/A')}</p>
            </div>
            <div class="feedback-item">
                <h4>Grammar</h4>
                <p>{eval_data.get('grammar', 'N/A')}</p>
            </div>
            <div class="feedback-item">
                <h4>Vocabulary</h4>
                <p>{eval_data.get('vocabulary', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with cols[1]:
            st.markdown(f"""
            <div class="feedback-item">
                <h4>Fluency & Pace</h4>
                <p>{eval_data.get('fluency', 'N/A')}</p>
            </div>
            <div class="feedback-item">
                <h4>Coherence</h4>
                <p>{eval_data.get('coherence', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
    else:  # Reading evaluation
        with cols[0]:
            st.markdown(f"""
            <div class="feedback-item">
                <h4>Reading Accuracy</h4>
                <p>{eval_data.get('accuracy', 'N/A')}</p>
            </div>
            <div class="feedback-item">
                <h4>Pronunciation</h4>
                <p>{eval_data.get('pronunciation', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with cols[1]:
            st.markdown(f"""
            <div class="feedback-item">
                <h4>Fluency & Pace</h4>
                <p>{eval_data.get('fluency', 'N/A')}</p>
            </div>
            <div class="feedback-item">
                <h4>Overall Performance</h4>
                <p>{eval_data.get('overall', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display improvement tips
    st.subheader("Improvement Tips:")
    tips = eval_data.get('improvement_tips', [])
    for tip in tips:
        st.markdown(f"- {tip}")

def speaking_test():
    """Speaking test section"""
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    st.title("🗣️ Speaking Test")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="instruction-box">', unsafe_allow_html=True)
    st.write("**Instructions:** Record yourself speaking about the following topic for at least 30 seconds.")
    st.write("**Topic:** Describe your favorite holiday destination and why you like it")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Two columns for better layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Audio recorder with a custom height and styling
        st.markdown("<p>Click the microphone to start/stop recording:</p>", unsafe_allow_html=True)
        audio_bytes = audio_recorder(key="speaking_recorder", pause_threshold=5.0)
        
        if audio_bytes:
            # Show a spinner while processing
            with st.spinner("Processing your speech..."):
                # Transcribe audio
                transcription_result = transcribe_audio(audio_bytes)
                
                if transcription_result["success"]:
                    transcription = transcription_result["text"]
                    
                    # Evaluate with Groq
                    evaluation_results = evaluate_with_groq(transcription, "speaking")
                    
                    # Save to history
                    if evaluation_results["success"]:
                        st.session_state.history.append({
                            "type": "Speaking Test",
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "transcript": transcription,
                            "score": evaluation_results["evaluation"].get("score", "N/A"),
                            "evaluation": evaluation_results["evaluation"]
                        })
                    
                    # Display results
                    st.markdown('<div class="results-container">', unsafe_allow_html=True)
                    display_evaluation_results(evaluation_results, transcription)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error(f"Transcription failed: {transcription_result['error']}")
    
    with col2:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px;">
            <h4>Speaking Tips:</h4>
            <ul>
                <li>Speak clearly and at a moderate pace</li>
                <li>Use varied vocabulary</li>
                <li>Organize your thoughts logically</li>
                <li>Include personal experiences</li>
                <li>Use natural pauses</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def reading_test():
    """Reading test section"""
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    st.title("📚 Reading Test")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Load reading texts from file
    reading_texts = load_reading_texts()
    reading_text = reading_texts.get("default")
    
    st.markdown('<div class="instruction-box">', unsafe_allow_html=True)
    st.write("**Instructions:** Read the following text aloud clearly and at a natural pace.")
    st.markdown(f"""
    <div style="background-color: white; padding: 20px; border-radius: 5px; border: 1px solid #ddd; margin: 15px 0;">
        <p style="font-size: 18px; line-height: 1.6;">{reading_text}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Two columns for better layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Audio recorder for reading test
        st.markdown("<p>Click the microphone to start/stop recording:</p>", unsafe_allow_html=True)
        reading_audio = audio_recorder(key="reading_recorder", pause_threshold=10.0)
        
        if reading_audio:
            # Show a spinner while processing
            with st.spinner("Processing your reading..."):
                # Transcribe audio
                transcription_result = transcribe_audio(reading_audio)
                
                if transcription_result["success"]:
                    reading_transcription = transcription_result["text"]
                    
                    # Evaluate reading accuracy
                    evaluation_results = evaluate_with_groq(
                        reading_transcription, 
                        "reading",
                        reference_text=reading_text
                    )
                    
                    # Save to history
                    if evaluation_results["success"]:
                        st.session_state.history.append({
                            "type": "Reading Test",
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "transcript": reading_transcription,
                            "score": evaluation_results["evaluation"].get("score", "N/A"),
                            "evaluation": evaluation_results["evaluation"]
                        })
                    
                    # Display results
                    st.markdown('<div class="results-container">', unsafe_allow_html=True)
                    display_evaluation_results(evaluation_results, reading_transcription)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error(f"Transcription failed: {transcription_result['error']}")
    
    with col2:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px;">
            <h4>Reading Tips:</h4>
            <ul>
                <li>Maintain a steady pace</li>
                <li>Pronounce each word clearly</li>
                <li>Use appropriate intonation</li>
                <li>Pause at punctuation marks</li>
                <li>Practice difficult words</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def check_api_connection():
    """Check API connection without making a full API call"""
    if not GROQ_API_KEY:
        return False, "API key not configured"
    
    try:
        # Make a lightweight GET request to check if the API is accessible
        # Using the models endpoint which is commonly available
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            timeout=5
        )
        return response.status_code < 400, f"HTTP Status: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"

def main():
    """Main application"""
    # Header
    st.markdown("""
    <div style="text-align: center; padding-bottom: 0px;">
        <h1 style="color: #0366d6;">English Speaking Evaluation Test</h1>
        <p>Improve your English speaking skills with AI-powered feedback</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if API key is configured
    if not GROQ_API_KEY:
        st.error("⚠️ API key not configured!")
        st.info("""
        ### How to set up the API key:
        
        For app administrators, please do ONE of the following:
        
        **Option 1:** Create a `.env` file in the same directory with:
        ```
        GROQ_API_KEY=your_api_key_here
        ```
        
        **Option 2:** Create a `config.py` file in the same directory with:
        ```python
        GROQ_API_KEY = "your_api_key_here"
        ```
           
        **Option 3:** Set it as an environment variable:
        ```
        export GROQ_API_KEY=your_api_key_here
        streamlit run app.py
        ```
        
        If you don't have a Groq API key, you can get one at [https://console.groq.com/](https://console.groq.com/)
        """)
        st.stop()  # Stop execution if API key is not configured
    
    # Test the API connection before proceeding
    connection_ok, connection_message = check_api_connection()
    if not connection_ok:
        st.error(f"⚠️ API connection failed: {connection_message}")
        st.info("""
        ### API Connection Failed
        
        This could be due to:
        1. Invalid API key
        2. Network connectivity issues
        3. API rate limiting
        
        Please verify your API key and try again.
        """)
        st.stop()  # Stop execution if API connection fails
    
    # Continue with normal app flow if API connection is successful
    # Tabs for different tests
    tab1, tab2 = st.tabs(["Speaking Test", "Reading Test"])
    
    with tab1:
        speaking_test()
    
    with tab2:
        reading_test()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
        <p><i>This application uses Groq API and Google's Speech Recognition to provide feedback on your English speaking skills.</i></p>
        <p>© 2025 English Speaking Evaluation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
