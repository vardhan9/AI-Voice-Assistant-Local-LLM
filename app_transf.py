import os
import base64
import subprocess
import platform
import tempfile
import whisper
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from pydub import AudioSegment

# Initialize the model and tokenizer
MODEL_PATH = "data/transformer_models/llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)

# Tie weights and initialize lm_head if necessary
if not model.config.tie_word_embeddings:
    model.tie_weights()

# Audio placeholder for playback
audio_placeholder = st.empty()

import os
import subprocess
import platform
import tempfile

def combine_audio_files(file_list, output_file):
    """Combine multiple audio files into one."""
    combined = AudioSegment.empty()
    for file in file_list:
        if file.endswith("temp_audio.wav"):  # Skip 'temp_audio.wav'
            continue
        audio = AudioSegment.from_wav(file)
        combined += audio  # Add each file to the combined output
    
    combined.export(output_file, format="wav")
    print(f"Combined audio saved to {output_file}")

# Cleanup temporary files
def cleanup_wav_files_in_current_directory():
    try:
        # Get the current working directory where the script is located
        current_directory = os.getcwd()
        
        # Loop through all files in the directory
        for file_name in os.listdir(current_directory):
            if file_name.endswith(".wav"):
                file_path = os.path.join(current_directory, file_name)
                
                # Delete the .wav file
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        
        print("Cleanup complete.")
    
    except Exception as e:
        print(f"Error cleaning up .wav files: {e}")

def speak_text_with_piper(text_to_speak, voice_folder="piper_tts/voices/default_female_voice"):
    """Generate speech from text using Piper TTS."""
    # Determine the Piper binary
    operating_system = platform.system()
    piper_binary = os.path.join("piper_tts", "piper.exe" if operating_system == "Windows" else "piper")

    # Construct the voice path and check if it exists
    voice_path = os.path.abspath(voice_folder)

    # Locate the model and JSON files
    model_path = next((os.path.join(voice_path, f) for f in os.listdir(voice_path) if f.endswith('.onnx')), None)
    json_path = next((os.path.join(voice_path, f) for f in os.listdir(voice_path) if f.endswith('.json')), None)

    if not model_path or not json_path:
        print("Required model or JSON files are missing.")
        return None

    try:
        temp_output_path = "temp_speech.wav"  # Ensure this is a single fixed file name
    
        # Construct the Piper command
        command = [
            piper_binary,
            "-m", model_path,
            "-c", json_path,
            "-o", temp_output_path,
        ]

        # Run the Piper command and capture the output and error messages
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(text_to_speak.encode("utf-8"))
        process.wait()

        # Assuming Piper creates multiple .wav files
        file_list = [f for f in os.listdir() if f.endswith(".wav")]
        combine_audio_files(file_list, "final_combined_speech.wav")

        audio_placeholder.markdown(
            f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{base64.b64encode(open("final_combined_speech.wav", "rb").read()).decode()}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            """,
            unsafe_allow_html=True,
        )

        cleanup_wav_files_in_current_directory()
        
       
    except Exception as e:
        print(f"Error running Piper TTS: {e}")
        return None


# Recognize speech from an audio file
def recognize_speech_whisper(audio_file=None, model_size="base"):
    if audio_file:
        try:
            # Load the specified Whisper model
            model = whisper.load_model(model_size)
            
            # Perform transcription with Finnish language detection
            result = model.transcribe(audio_file)
            
            # Print and return the detected speech
            detected_speech = result["text"]
            print(f"Detected speech: {detected_speech}")
            return detected_speech
        except Exception as e:
            print(f"Error recognizing speech: {e}")
            return None
    else:
        print("No audio file provided.")
        return None


# Generate a response using the model
def process_prompt(prompt):
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    )

    # Set the pad_token_id explicitly if it's missing
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    outputs = model.generate(
        inputs["input_ids"],
        max_length=500,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Save conversation history to a file
def save_conversation(messages, filename="conversation_log.txt"):
    with open(filename, "w") as file:
        for message in messages:
            role = "User" if message["role"] == "user" else "Assistant"
            file.write(f"{role}: {message['content']}\n")



# Handle Streamlit interface
def handle_interface():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Ask me a question!"}]

    col1, col2, col3 = st.columns(3)
    audio_file = None
    audio_prompt = None

    with col2:
        # Record audio directly without saving to a file
        audio_file = audio_recorder()
        if audio_file:
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_file)
            audio_prompt = recognize_speech_whisper("temp_audio.wav")
           

    # Add a text input for entering the prompt
    #text_prompt = st.text_input("Enter your question:", "")

    # Combine prompts from text and audio
    #prompt = text_prompt or audio_prompt
    prompt = audio_prompt
    

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            response = process_prompt(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            print(response)
            # Generate TTS audio and play it directly in Streamlit
            audio_data = speak_text_with_piper(response)
            if audio_data:
                # Play the audio using Streamlit's audio player directly from the audio data
                #st.audio(audio_data, format="audio/wav")
                # Play audio using Streamlit's st.audio
                audio_placeholder.markdown(
            f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{base64.b64encode(open(audio_data, "rb").read()).decode()}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            """,
            unsafe_allow_html=True,
        )
            else:
                st.error("Failed to generate audio.")

    # Display conversation history
    for message in reversed(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])

    save_conversation(st.session_state.messages)

# Main function
def main():
    st.title("AI Voice Assistant - Local Transformers Model")
    handle_interface()

if __name__ == "__main__":
    main()
