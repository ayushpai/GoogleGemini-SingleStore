import os
import os
import google.generativeai as genai
from IPython.display import Markdown
from IPython.display import display
import PIL.Image
import time

os.environ['GOOGLE_API_KEY'] = "YOUR_KEY" # input your key here
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
vision_model = genai.GenerativeModel('gemini-pro-vision')


def analyze_image():
    image = PIL.Image.open('frames/frame.jpg')
    response = vision_model.generate_content(
        ["Please output 2 sentences about what you see in this video frame.", image])
    return response.text


def main():

    while True:
        # analyze posture
        print("👀 Gemini is watching...")
        analysis = analyze_image()
        print("🎙️ Gemini says:")
        print(analysis)

        # wait for 5 seconds
        time.sleep(5)


if __name__ == "__main__":
    main()
