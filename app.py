
####################################################
import streamlit as st # Streamlit is used to build interactive web apps in Python. st is the commonly used alias.
import os # Lets you interact with the operating system, like accessing environment variables or file paths.
import time # To manage delays, measure performance duration, or create wait periods.
import json # To handle JSON data, such as reading/writing configuration files, saving results, or parsing API responses.
import tempfile #Used to create temporary files (e.g., saving audio files from mic or upload).
from datetime import datetime # To work with dates and times — useful for timestamping test attempts or logging activity.
from groq import Groq # To interact with the Groq AI API, which can run large language models like LLaMA3 at high speed
import speech_recognition as sr # To transcribe audio into text using various speech recognition engines (e.g., Google, Sphinx).
from audio_recorder_streamlit import audio_recorder # Imports the audio_recorder function from the audio_recorder_streamlit package. 




# Set page configuration
st.set_page_config(# Sets the title shown in the browser tab and centers the layout for a clean look.
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
    st.session_state.history = [] # initializes an empty list to store the user’s previous test attempts or 
                                # interactions (e.g., timestamps, transcripts, scores).

if 'current_test' not in st.session_state: # Checks if the session has a current_test already initialized.
    st.session_state.current_test = None # Initializes current_test as None, to later store the test metadata (like question set, difficulty, etc.).

if 'api_key' not in st.session_state: # Checks if an API key for the Groq API is already stored in the session state.
    st.session_state.api_key = os.environ.get("GROQ_API_KEY", "") # tries to fetch the key from the system environment variable GROQ_API_KEY,
                                                                  # and sets it in session state.

# Sidebar for configuration
with st.sidebar: # anything inside this block appears on the left-hand sidebar of the app.
    st.image("https://via.placeholder.com/150x150.png?text=E-Speak", width=150) # Displays a placeholder image (logo for your app — "E-Speak") and the title "Settings".
    st.title("Settings")
    
    # API configuration
    api_key_input = st.text_input("Groq API Key", value=st.session_state.api_key, type="password") # Adds a password-protected input box for the user to paste their Groq API key.
    if api_key_input != st.session_state.api_key: # If the entered key is different from the one already stored in session_state, it updates the stored key.
        st.session_state.api_key = api_key_input
    
    st.markdown("---")
    
    # Model selection
    model_option = st.selectbox( # Adds a dropdown menu (selectbox) to choose one of several large language models (LLMs).
        "LLM Model",
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
        index = 0
    )
        # Default selection is the first model: "llama3-8b-8192".
        # #These options refer to different Groq-supported models:
        # #LLaMA3 8B – Lightweight, fast.
        #LLaMA3 70B – Larger, more accurate.
        # #Mixtral – Mixture-of-experts model, capable of advanced reasoning.
        
    
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    #Adds a slider to control temperature, a parameter that affects the creativity of AI-generated responses:
    # 0.0 = deterministic (always same output)
    # 1.0 = highly creative/random
    # Useful for controlling how strict or freeform the AI’s responses are during evaluations.
    
    st.markdown("---") # Just adds a horizontal rule to separate sections in the sidebar for cleaner visuals.
    
    # History section
    # This block displays a summary of past test attempts by the user inside the sidebar or main layout (depending on where it's placed).
    
    st.subheader("Test History") # Displays a subheader titled "Test History" 
    if st.session_state.history:
        for i, entry in enumerate(st.session_state.history):
            with st.expander(f"{entry['type']} - {entry['date']}"): # Allows user to collapse/expand each test result
                st.write(f"**Score:** {entry['score']}/10") # Shows test performance 
                st.write(f"**Transcript:** {entry['transcript'][:100]}...") # shows transcript preview
    else:
        st.info("No test history yet")

# Initialize Groq client
@st.cache_resource
def get_groq_client(api_key):
    if not api_key:
        return None
    return Groq(api_key=api_key)

client = get_groq_client(st.session_state.api_key)

def transcribe_audio(audio_bytes):
    """Convert speech to text using SpeechRecognition"""
    # Create a temporary file for the audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name
    
    # Transcribe the audio
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                result = {"success": True, "text": text}
            except sr.UnknownValueError:
                result = {"success": False, "error": "Could not understand audio"}
            except sr.RequestError:
                result = {"success": False, "error": "Speech recognition API unavailable"}
    except Exception as e:
        result = {"success": False, "error": f"Error processing audio: {str(e)}"}
    
    # Clean up the temporary file
    try:
        os.unlink(temp_audio_path)
    except:
        pass
        
    return result

def evaluate_with_groq(text, evaluation_type="speaking", reference_text=None):
    """Get evaluation from Groq API"""
    if not client:
        return {"success": False, "error": "Groq API key is not configured"}
    
    try:
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
        else:  # reading evaluation
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

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            model=model_option,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return {"success": True, "evaluation": result}
    
    except Exception as e:
        return {"success": False, "error": f"Error from Groq API: {str(e)}"}

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
    
    reading_text = """The rapid advancement of artificial intelligence has brought significant changes 
            to various industries. While some fear job displacement, others believe AI will create new 
            opportunities and enhance human capabilities. Researchers continue to debate the long-term 
            implications of these technologies on society, economy, and human cognition."""
    
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

def main():
    """Main application"""
    # Header
    st.markdown("""
    <div style="text-align: center; padding-bottom: 30px;">
        <h1 style="color: #0366d6;">📣 English Speaking Evaluation Test 📣</h1>
        <p>Improve your English speaking skills with AI-powered feedback</p>
    </div>
    """, unsafe_allow_html=True)
    

        # Check if API key is configured
    if not st.session_state.api_key:
        st.warning("⚠️ Please enter your Groq API key in the sidebar to use this application.")
        st.info("If you don't have a Groq API key, you can get one at https://console.groq.com/")
        st.stop()
    
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

    # https://github.com/darigain/fluency
    # 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
    # https://www.fluencyflow.ai/


if __name__ == "__main__":
    main()
