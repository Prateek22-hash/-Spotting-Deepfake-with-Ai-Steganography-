import cv2
import numpy as np

import os

def encode_char_in_video(input_video_path, output_video_path, char='#'):
    # Check if input video exists
    if not os.path.exists(input_video_path):
        print(f"ERROR: Input video file does not exist: {input_video_path}")
        return False

    # Convert char to binary string
    char_bin = format(ord(char), '08b')
    char_len = len(char_bin)
    bit_idx = 0

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"ERROR: Failed to open input video: {input_video_path}")
        return False

    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    
    if not output_video_path.lower().endswith('.avi'):
        output_video_path = output_video_path.rsplit('.', 1)[0] + '.avi'

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"ERROR: Failed to open output video for writing: {output_video_path}")
        cap.release()
        return False

    first_frame_saved = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if bit_idx < char_len:
            for i in range(8):
                x = i % width
                y = i // width
                b, g, r = frame[y, x]
                r = (int(r) & 0xFE) | int(char_bin[bit_idx])
                frame[y, x] = (b, g, r)
                bit_idx += 1
                if bit_idx >= char_len:
                    break

            if not first_frame_saved:
                cv2.imwrite('debug_first_frame.png', frame)
                first_frame_saved = True

        out.write(frame)

    cap.release()
    out.release()
    return True

def decode_char_from_video(video_path):
    import os
    if not os.path.exists(video_path):
        print(f"ERROR: Video file does not exist: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Failed to open video file: {video_path}")
        return None

    bits = []
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video frame")
        cap.release()
        return None

    
    for i in range(8):
        x = i % frame.shape[1]
        y = i // frame.shape[1]
        b, g, r = frame[y, x]
        bits.append(str(r & 1))

    print(f"DEBUG: Extracted bits from video: {bits}")  # Debug print bits
    print("Video converted and added to the model training dataset and ready to test if all the bits are 1")
    cap.release()

    char_bin = ''.join(bits)
    char = chr(int(char_bin, 2))
    return char

if __name__ == "__main__":
    # Example usage
    encode_char_in_video(r'C:\Users\bajaj\PycharmProjects\PythonProject\04.avi', '004.avi')
    decode_char_from_video('004.avi')
