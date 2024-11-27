import os
import glob
import pygame
import pyttsx3
import speech_recognition as sr
import streamlit as st
from gtts import gTTS
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import time
# Initialize Ollama model
llm = Ollama(model="llama3.2:1b", base_url="http://127.0.0.1:11434", verbose=True)

# Helper function to speak text using gTTS and pygame
def speak_text(text, lang="fi"):
    temp_file = "temp_speech.mp3"
    try:
        # Generate speech as an MP3 file
        tts = gTTS(text=text, lang=lang)
        tts.save(temp_file)

        # Initialize pygame mixer and play the audio
        pygame.mixer.init()
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()

        # Wait for the speech to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    finally:
        # Cleanup temporary file after playback
        pygame.mixer.music.stop()
        pygame.mixer.quit()  # Explicitly close the mixer
        # Allow time for the music to fully stop and the file to be released
        time.sleep(0.5)  # Add a small delay to ensure proper file release
        cleanup_file(temp_file)
        print("Speech playback finished.")

# Cleanup function to delete temporary .mp3 files
def cleanup_file(file_path):
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Deleted {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
    else:
        print(f"File {file_path} not found.")

# Function to clear all .mp3 files from a specified directory
def clear_mp3_files(directory):
    mp3_files = glob.glob(os.path.join(directory, "*.mp3"))
    if mp3_files:
        for file in mp3_files:
            os.remove(file)
            print(f"Deleted: {file}")
    else:
        print("No .mp3 files found in the directory.")

# Send a prompt to the Ollama model and get the response
def send_prompt(prompt):
    response = llm.invoke(prompt)
    speak_text(response)  # Speak the response
    return response

# Speech recognition function
def recognize_speech():
    r = sr.Recognizer()
    listening = st.empty()
    with sr.Microphone() as source:
        listening.markdown("I am listening....")
        audio_text = r.listen(source)
        listening.markdown("Time over, thanks")
        
    try:
        recognized_text = r.recognize_google(audio_text, language='fi-FI')
        st.write(f"Sinä kysyi: {recognized_text}")
        return recognized_text
    except:
        st.write("Sorry, I did not get that")
        return None

# Text-to-Speech with pyttsx3
def text_to_speech(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 100)  # Speed of speech
        engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.write(f"Error in speech synthesis: {e}")

# Streamlit interface and button handling
def handle_buttons():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Ask me a question!"}]

    prompt = None

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        talk = st.button('Talk', use_container_width=True)


    if talk:
        prompt = recognize_speech() or "Sorry, I didn't get that"



    # Handle responses
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = send_prompt(prompt)
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Main function to run Streamlit interface
def main():
    st.title("AI - Local Assistant 100%")
    handle_buttons()

if __name__ == "__main__":
    main()
