import wave
import numpy as np

def encode_char_in_audio(input_audio_path, output_audio_path, char='#'):
    # Convert char to binary string
    char_bin = format(ord(char), '08b')

    # Open the audio file
    with wave.open(input_audio_path, 'rb') as audio:
        params = audio.getparams()
        frames = audio.readframes(params.nframes)
        audio_data = np.frombuffer(frames, dtype=np.int16).copy()  # Make writable copy

    # Modify the LSB of the first 8 samples to encode the char bits
    for i in range(8):
        audio_data[i] = (audio_data[i] & ~1) | int(char_bin[i])

    # Write the modified audio data to output file
    with wave.open(output_audio_path, 'wb') as audio_out:
        audio_out.setparams(params)
        audio_out.writeframes(audio_data.tobytes())

    print("Video converted and added to the model training dataset and ready to test")

def decode_char_from_audio(audio_path):
    with wave.open(audio_path, 'rb') as audio:
        frames = audio.readframes(audio.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16)

    bits = []
    for i in range(8):
        bits.append(str(audio_data[i] & 1))

    char_bin = ''.join(bits)
    char = chr(int(char_bin, 2))
    return char

if __name__ == "__main__":
    # Example usage
    encode_char_in_audio(r'C:\Users\bajaj\Downloads\w].wav', 'Women2.wav')
    decode_char_from_audio('Women2.wav')
