import os
import base64
import streamlit as st
from gtts import gTTS
from transformers import AutoModelForCausalLM, AutoTokenizer
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
import pyttsx3

# Initialize the model and tokenizer
MODEL_PATH = "data/transformer_models/llama-3.2-3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
# Tie weights and initialize lm_head if necessary
if not model.config.tie_word_embeddings:
    model.tie_weights()
# Audio placeholder for playback
audio_placeholder = st.empty()

# Generate and play speech using gTTS
def speak_text(text, lang="fi"):
    temp_file = "temp_speech.mp3"
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save(temp_file)

        # Streamlit audio playback
        with open(temp_file, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode()
        audio_placeholder.markdown(
            f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{audio_data}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            """,
            unsafe_allow_html=True,
        )
    finally:
        cleanup_file(temp_file)

# Delete temporary files
def cleanup_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

# Recognize speech from an audio file
def recognize_speech(audio_file=None):
    r = sr.Recognizer()
    if audio_file:
        try:
            with sr.AudioFile(audio_file) as source:
                audio = r.record(source)
                st.write('Detected speech:',r.recognize_google(audio, language='fi-FI'))
            return r.recognize_google(audio, language="fi-FI")
        except Exception as e:
            st.error(f"Error recognizing speech: {e}")
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

    # Set the `pad_token_id` explicitly if it's missing
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    outputs = model.generate(
        inputs["input_ids"],
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True,
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
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
        audio_file = audio_recorder()
        if audio_file:
            with open("temp_audio.wav", "wb") as f:
                f.write(audio_file)
            audio_prompt = recognize_speech("temp_audio.wav")
            cleanup_file("temp_audio.wav")

    # Add a text input for entering the prompt
    text_prompt = st.text_input("Enter your question:", "")

    # Combine prompts from text and audio
    prompt = text_prompt or audio_prompt

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            response = process_prompt(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            speak_text(response)

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
