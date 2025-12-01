from PIL import Image

def encode_char_in_image(input_image_path, output_image_path, char='#'):
    # Open the image
    img = Image.open(input_image_path)
    img = img.convert('RGB')
    pixels = img.load()

    # Convert char to binary string
    char_bin = format(ord(char), '08b')

    width, height = img.size
    idx = 0

    # Encode the character bits into the least significant bit of the red channel of pixels
    for y in range(height):
        for x in range(width):
            if idx < len(char_bin):
                r, g, b = pixels[x, y]
                # Modify the LSB of red channel
                r = (r & ~1) | int(char_bin[idx])
                pixels[x, y] = (r, g, b)
                idx += 1
            else:
                break
        if idx >= len(char_bin):
            break

    img.save(output_image_path)
    print("Image converted and added to the model training dataset and ready to test")

def decode_char_from_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    pixels = img.load()

    width, height = img.size
    bits = []

    # Extract the LSB of red channel from first 8 pixels
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            bits.append(str(r & 1))
            if len(bits) == 8:
                break
        if len(bits) == 8:
            break

    char_bin = ''.join(bits)
    char = chr(int(char_bin, 2))
    return char

if __name__ == "__main__":
    # Example usage
    encode_char_in_image(r'C:\Users\bajaj\PycharmProjects\PythonProject\static\bounding box\fake\1010.jpg', 'encoded_image.png')
    decode_char_from_image('encoded_image.png')
