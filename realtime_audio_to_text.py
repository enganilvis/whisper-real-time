import argparse
import io
import wave
from flask import Flask, request, jsonify
import speech_recognition as sr
import numpy as np
import torch
import whisper
from pydub import AudioSegment

app = Flask(__name__)

# Initialize the recognizer
recognizer = sr.Recognizer()

@app.route('/api/audio/upload', methods=['POST'])
def upload_audio():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)

    args = parser.parse_args()

    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    audio_model = whisper.load_model(model)

    # Ensure request has audio data
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio part in the request'}), 400

    audio_file = request.files['audio']

    # Check if the file is empty
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400

    # Check if the file is a valid audio file
    if audio_file:
        try:
            print(audio_file.filename);
            
            audio_data = audio_file.read()
            print(audio_data[:4])
            
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))

            # Export AudioSegment as WAV format (16-bit PCM)
            audio_wav = audio_segment.export(format='wav')

            # Export AudioSegment as WAV format (16-bit PCM)
            wav_buffer = io.BytesIO()
            audio_segment.export(wav_buffer, format='wav')

            # Reset buffer position to start
            wav_buffer.seek(0)

            # Get sample width and frame rate using wave module
            with wave.open(wav_buffer, 'rb') as wav_file:
             sample_width = wav_file.getsampwidth()
             frame_rate = wav_file.getframerate()

             # Read the entire audio content as bytes
            audio_content = wav_file.readframes(wav_file.getnframes())

            print(sample_width)
            print(frame_rate)

            # Convert raw audio content to NumPy array
            audio_np_data = np.frombuffer(audio_content, dtype=np.int16).astype(np.float32) / 32768.0
 
            audio_data = sr.AudioData(audio_wav.read(), sample_width=sample_width, sample_rate=frame_rate // sample_width)
            
            # Convert in-ram buffer to something the model can use directly without needing a temp file.
            # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
            # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
            audio_np = np.frombuffer(audio_np_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Read the transcription.
            result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
            text_output = result['text'].strip()

            return jsonify({'transcription': text_output})
        except sr.UnknownValueError:
            return jsonify({'error': 'Speech recognition could not understand audio'}), 500
        except sr.RequestError as e:
            return jsonify({'error': f'Speech recognition service error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True,port=5000)
