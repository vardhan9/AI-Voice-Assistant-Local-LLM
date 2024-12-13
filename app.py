import os
import glob
import pygame
import time
import streamlit as st
from gtts import gTTS
from langchain_community.llms import Ollama
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
import base64
# Initialize Ollama model
llm = Ollama(model="llama3.2:1b", base_url="http://127.0.0.1:11434", verbose=True)
audio_placeholder = st.empty()

# Speak text using gTTS and pygame
def speak_text(text, lang="fi"):
    temp_file = "temp_speech.mp3"
    try:
        # Generate and save speech
        tts = gTTS(text=text, lang=lang)
        tts.save(temp_file)

        # Play audio using Streamlit's st.audio
        audio_placeholder.markdown(
            f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{base64.b64encode(open(temp_file, "rb").read()).decode()}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            """,
            unsafe_allow_html=True,
        )
    finally:
        cleanup_file(temp_file)

# Cleanup temporary files
def cleanup_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

# Recognize speech from audio
def recognize_speech(audio_file=None):
    r = sr.Recognizer()
    if audio_file:
        try:
            with sr.AudioFile(audio_file) as source:
                audio = r.record(source)
                st.write('Detected spech:',r.recognize_google(audio, language='fi-FI'))
            return r.recognize_google(audio, language='fi-FI')
        except Exception as e:
            print(f"Error recognizing speech: {e}")
    return None

# Process user prompt with Ollama and respond
def process_prompt(prompt):
    response = llm.invoke(prompt)
    speak_text(response)  # Speak the response
    return response

# Save conversation to file
def save_conversation(messages, filename="conversation_log.txt"):
    with open(filename, "w") as file:
        for message in messages:
            role = "User" if message["role"] == "user" else "Assistant"
            file.write(f"{role}: {message['content']}\n")
        file.write("\n")

# Streamlit UI
def handle_interface():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Ask me a question!"}]

    col1, col2, col3 = st.columns(3)
    audio_file = None
    prompt = None

    with col2:
        audio_file = audio_recorder()
        if audio_file:
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_file)
            prompt = recognize_speech("temp_audio.wav")
            cleanup_file("temp_audio.wav")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            response = process_prompt(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Display conversation history
    for message in reversed(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
    save_conversation(st.session_state.messages)

# Main function
def main():
    st.title("AI Voice Assistant - Local LLM")
    handle_interface()

if __name__ == "__main__":
    main()
