from flask import Flask, request, jsonify
import whisper
import numpy as np
import torch
from pydub import AudioSegment
import io
import re
from word2number import w2n

app = Flask(__name__)

# Load the Whisper model (adjust model name as needed)
model_name = "base.en"  # Default to English model
audio_model = whisper.load_model(model_name)

def convert_words_to_numbers_and_mask(text):
    # Define a function to replace and mask account numbers
    def replace_and_mask_accounts(match):
        return f"{match.group(1)}****{match.group(2)}"
    
    # Define a function to replace and mask mobile numbers
    def replace_and_mask_mobiles(match):
        return f"{match.group(1)}****{match.group(2)}"
    
    # Regular expression pattern to find account numbers (e.g., 1234 5678 9012)
    account_pattern = re.compile(r'(\b\d{4})\s*(\d{4})\s*(\d{4}\b)')
    
    # Regular expression pattern to find mobile numbers (e.g., 123 456 7890)
    mobile_pattern = re.compile(r'(\b\d{3})\s*(\d{3})\s*(\d{4}\b)')
    
    # Convert words to numbers
    try:
        # Split text into words and attempt to convert each word to a number
        words = text.split()
        converted_words = []
        for word in words:
            try:
                converted_word = str(w2n.word_to_num(word))
            except ValueError:
                # If word cannot be converted, keep it as is
                converted_word = word
            converted_words.append(converted_word)
        
        # Join converted words back into text
        text = ' '.join(converted_words)
    except Exception as e:
        return f"Error converting words to numbers: {str(e)}"
    
    # Mask account numbers
    text = account_pattern.sub(replace_and_mask_accounts, text)
    
    # Mask mobile numbers
    text = mobile_pattern.sub(replace_and_mask_mobiles, text)
    
    return text

@app.route('/transcribe', methods=['POST'])
def transcribe():
    global audio_model

    try:
        # Check if 'audio' file part is present in the request
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file part in the request'}), 400

        audio_file = request.files['audio']
        
        # Optionally, retrieve language parameter from request form data
        language = request.form.get('language', 'en')  # Default to English if not specified
        
        # Read the audio file using pydub
        audio = AudioSegment.from_file(io.BytesIO(audio_file.read()))
        
        # Ensure the audio is in the correct format (mono, 16kHz)
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Convert audio to numpy array
        audio_np = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        
        # Perform transcription using Whisper model with specified language
        result = audio_model.transcribe(audio_np, language=language, fp16=torch.cuda.is_available())
        transcript = result['text'].strip()

        masked_transcript = convert_words_to_numbers_and_mask(transcript)

        print(masked_transcript)

        return jsonify({'transcript': masked_transcript})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True,port=5000)
