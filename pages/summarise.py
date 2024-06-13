
# import modules
import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import BartForConditionalGeneration, BartTokenizer
from gtts import gTTS
import os

# streamlit
st.set_page_config(page_title="TTS Summariser", page_icon="💬")
st.title("Webpage Summariser with TTS")
url = st.text_input("Enter URL of the webpage:")
summarise_button = st.button("Summarise")

# fetch website content
def find_main_content(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}, stream=True)

        if response.status_code != 200:
            st.error(f"Failed to fetch content from {url}. Response code is {response.status_code}")
            return None

        with st.expander("Full content of website"):
            st.write(response.text)
        soup = BeautifulSoup(response.text, 'html.parser')
        main_content = ""
        for tag in soup.find_all(['p', 'article', 'div']):
            text = tag.get_text(strip=True)
            if text:
                main_content += text + "\n"

        return main_content.strip()
    except Exception as e:
        st.error(f"An error occurred while fetching content from {url}: {e}")
        return None

# summarise content
def summarise_content(content):
    try:
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

        inputs = tokenizer([content], max_length=1024, truncation=True, return_tensors='pt')
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, min_length=50, max_length=500, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary
    except Exception as e:
        st.error(f"An error occurred while summarizing content: {e}")
        return None

# convert to TTS
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts_file = "speech.mp3"
        tts.save(tts_file)
        return tts_file
    except Exception as e:
        st.error(f"An error occurred while converting text to speech: {e}")
        return None

# streamlit
if summarise_button:
    main_content = find_main_content(url)
    page_content = "test"
    if main_content:
        summarised_content = summarise_content(main_content)
        if summarised_content:
            st.write("Summarised Content:")
            st.write(summarised_content)

            st.write("Playing Summarised Content (Text-to-Speech):")
            tts_file = text_to_speech(summarised_content)
            if tts_file:
                st.audio(tts_file, format='audio/mp3')

                os.remove(tts_file)
