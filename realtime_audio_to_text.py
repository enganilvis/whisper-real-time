from flask import Flask, request, jsonify
import whisper
import numpy as np
import torch
from pydub import AudioSegment
import io
import re

app = Flask(__name__)

# Load the Whisper model (adjust model name as needed)
model_name = "base.en"  # Default to English model
audio_model = whisper.load_model(model_name)

# Function to mask mobile numbers
def mask_mobile_number(text):
    pattern = re.compile(r'(\b\d{3})\d{4}(\d{3}\b)')
    return pattern.sub(r'\1****\2', text)

# Function to mask account numbers
def mask_account_number(text):
    pattern = re.compile(r'(\b\d{4})\d{4}(\d{4}\b)')
    return pattern.sub(r'\1****\2', text)

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

        masked_transcript = mask_mobile_number(transcript)
        masked_transcript = mask_account_number(masked_transcript)

        return jsonify({'transcript': masked_transcript})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True,port=5000)
