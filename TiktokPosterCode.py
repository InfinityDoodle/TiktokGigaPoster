#https://ads.tiktok.com/business/creativecenter/inspiration/popular/hashtag/pc/en
import random
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import numpy as np
import soundfile as sf
import warnings
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from openai import OpenAI
import time
import os
import pyttsx3
from diffusers import StableDiffusionPipeline
import torch
from moviepy.editor import *
from moviepy.editor import ImageClip, VideoFileClip, concatenate_videoclips
import torch
from diffusers import StableDiffusion3Pipeline
from mutagen.mp3 import MP3
import re
from transformers import VitsModel, AutoTokenizer
import torch
import scipy
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips
from huggingface_hub import login
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
import time
from tiktok_uploader.browsers import get_browser
import selenium
from tiktok_uploader.auth import AuthBackend
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import selenium.common.exceptions
import tiktok_uploader
import tiktok_uploader.upload

# Replace YOUR_HUGGINGFACE_TOKEN with your actual token
login("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")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def postvid(description):
    # Setup Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")  # To avoid bot detection
    chrome_options.add_argument("start-maximized")

    # Path to your ChromeDriver executable
    driver = get_browser()

    # Go to TikTok's login page
    driver.get('https://www.tiktok.com/login')

    auth = AuthBackend(cookies=r"C:\Users\jgdiv\Downloads\www.tiktok.com_cookies.txt")

    driver = auth.authenticate_agent(driver)

    # Refresh the page to apply cookies
    driver.refresh()

    # Wait for some time to ensure login is completed
    time.sleep(10)


    # Navigate to the upload page after login
    driver.get('https://www.tiktok.com/upload')

    # Add delay to load the upload page
    time.sleep(12)

    # Locate the file input for video upload
    file_input = driver.find_element(By.CSS_SELECTOR, 'input[type="file"].jsx-399202212')

    # Use JavaScript to make the input field visible
    driver.execute_script("arguments[0].style.display = 'block';", file_input)

    # Provide the file path of the video you want to upload
    video_path = r'C:\Users\jgdiv\PycharmProjects\pythonProject\tiktokAutoPoster\output_video.mp4'
    file_input.send_keys(video_path)

    # Set the path to your video file

    # Upload the video

    # Add a delay to allow video to be uploaded
    time.sleep(15)

    # Locate the caption input and set the caption text
    caption_input = driver.find_element(By.XPATH, '//div[@contenteditable="true"]')
    caption_input.click()
    for _ in range(25):  # Adjust the range for how many characters you want to delete
        caption_input.send_keys(Keys.BACKSPACE)
    caption_input.send_keys(description)

    # Add delay if needed for reviewing the upload
    time.sleep(10)

    # Find the post button and click to submit the video

    tiktok_uploader.upload.post_video(driver)

    # Add a delay to ensure the post is processed
    time.sleep(20)

    # Close the browser
    driver.quit()


def getHashtags():
    url = 'https://ads.tiktok.com/business/creativecenter/inspiration/popular/hashtag/pc/en'

    # Headers to simulate a browser visit
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # Send request
    response = requests.get(url, headers=headers)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the section that contains hashtags and post counts
    hashtags = soup.find_all('span', "CardPc_titleText__RYOWo")
    hashlist = []
    for hash in hashtags:
        hashlist.append(hash.text)
        print(hash.text)
    return hashlist

def checkhashtagvalue(hashtag):
    genai.configure(api_key='PUT YOUR API KEY')
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("Using ONLY 0-100 please rate this theme on how easy it would be to make a ai generated video on. "
                                      "100 should be a theme that is not to broad, easy for ai to make a script, and"
                                      " would be good for making content. 0 is a something way to broad, not easily made with ai, and not good for making content. Please only use numbers that go from 0-100. Here is the theme. Even if you think doesnt mean anything or you want to make a comment please only use 0-100: " +hashtag)
    try:
        return (response.text)
    except:
        return 0


def clean_text(text):
    # Remove text within parentheses and square brackets, including the brackets themselves
    text_without_parentheses_and_brackets = re.sub(r'\(.*?\)|\[.*?\]', '', text)
    # Remove empty lines and strip leading/trailing whitespace from each line
    cleaned_lines = [line.strip() for line in text_without_parentheses_and_brackets.splitlines() if line.strip()]
    # Join cleaned lines back into a single string
    return '\n'.join(cleaned_lines)

def makeScript(hashtag):
    try:
        client = OpenAI(
            # This is the default and can be omitted
            api_key="PUT YOUR API KEY"
        )

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a expert tiktok script writer. You only give scripts that will be read out loud. This means not talking about the scene or what should be displayed."},
                {
                    "role": "user",
                    "content": "Please write a TikTok script based on the theme I’ll provide. The script should be 60 to 90 seconds long . starting with a strong hook that reappears later for continuity. It must be engaging, concise, and suitable for TikTok. Avoid false claims or personal anecdotes. Aim for informative content, and provide only the spoken script without any additional comments or directions. The content should be appropriate for all audiences and align with TikTok’s guidelines. Please do not talk about the scene. Only talk about the script. You also will not put # at the end. Here is the script: " + hashtag,
                }
            ],
            model="gpt-3.5-turbo"
            , stream=False
            , max_tokens=250
        )
        returnscript = clean_text(response.choices[0].message.content)
        print(returnscript)
        return (returnscript)
    except:
        client = OpenAI(
            # This is the default and can be omitted
            api_key="PUT YOUR API KEY"
        )
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a expert tiktok script writer. You only give scripts that will be read out loud. This means not talking about the scene or what should be displayed."},
                {
                    "role": "user",
                    "content": "Please write a TikTok script based on the theme I’ll provide. The script should be 60 to 90 seconds long . starting with a strong hook that reappears later for continuity. It must be engaging, concise, and suitable for TikTok. Avoid false claims or personal anecdotes. Aim for informative content, and provide only the spoken script without any additional comments or directions. The content should be appropriate for all audiences and align with TikTok’s guidelines. Please do not talk about the scene. Only talk about the script. You also will not put # at the end. Here is the script: " + hashtag,
                }
            ],
            model="gpt-3.5-turbo"
            , stream=False
            , max_tokens=250
        )
        returnscript = clean_text(response.choices[0].message.content)
        print(returnscript)
        return (returnscript)

def makeImage(prompt):
    try:
        genai.configure(api_key='PUT YOUR API KEY')
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content("I am about to give you a sentence. I want you to describe this sentence if it was a image. What you say will be used to generate a image. "
                                          "So, you need to make a detailed description of the image you think would fit this sentence."
                                          "For example if the sentence was a black cat on the side of the road you might respond with by describing the cat,"
                                          " the road, the houses on the road, the time of day, so on and so on. Try to keep it short and sweet. Also only respond "
                                          "with the description of the image you think would fit the sentence. Nothing else nothing more. Here is the sentence for you to use. It starts at this colon and end at the period: " + prompt + ". Please also don't describe any NSFW content. Make sure your response is below 77 characters."
                                                                                                                                                                             "Please dont expect anything like a name, date, or age to be inserted into the script later.")
        print(response.text)
    except:
        time.sleep(60)
        genai.configure(api_key='PUT YOUR API KEY')
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            "I am about to give you a sentence. I want you to describe this sentence if it was a image. What you say will be used to generate a image. "
            "So, you need to make a detailed description of the image you think would fit this sentence."
            "For example if the sentence was a black cat on the side of the road you might respond with by describing the cat,"
            " the road, the houses on the road, the time of day, so on and so on. Try to keep it short and sweet. Also only respond "
            "with the description of the image you think would fit the sentence. Nothing else nothing more. Here is the sentence for you to use. It starts at this colon and end at the period: " + prompt + ". Please also don't describe any NSFW content. Make sure your response is below 77 characters."
                                                                                                                                                                                                             "Please dont expect anything like a name, date, or age to be inserted into the script later.")

        print(response.text)
    # Load the model with torch_dtype set to float16 for better performance
    url = "https://api.getimg.ai/v1/stable-diffusion/text-to-image"

    payload = {
        "model": "synthwave-punk-v2",
        "width": 768,
        "height": 768,
        "steps": 90,
        "output_format": "png",
        "response_format": "url",
        "prompt": response.text
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": "Bearer PUT YOUR API KEY"
    }

    response = requests.post(url, json=payload, headers=headers)

    data = response.json()
    image_url = data.get("url")
    print(response.text)

    url = str(image_url)

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open a file in write-binary mode and save the image
        with open("generated_image_Tiktok.png", "wb") as file:
            file.write(response.content)
        print("Image downloaded successfully!")
    else:
        print(f"Failed to download the image. Status code: {response.status_code}")

def makeTTS(text, i):
    client = OpenAI(
        # This is the default and can be omitted
        api_key="PUT YOUR API KEY"
    )

    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text
    )

    with open(f"voiceForTiktok_{i}.wav", "wb") as f:
        f.write(response.content)

    print(f"voiceForTiktok_{i}.wav")

def split_string_at_nearest_space(text, max_length=30):
    # Split the text into words
    words = text.split()

    # List to hold the split lines
    lines = []
    current_line = ""

    for word in words:
        # Check if adding the next word exceeds the max length
        if len(current_line) + len(word) + 1 <= max_length:
            # Add the word to the current line
            current_line += word + " "
        else:
            # If the current line is full, save it and start a new line
            lines.append(current_line.strip())
            current_line = word + " "

    # Append the last line
    if current_line:
        lines.append(current_line.strip())

    return lines

    # Example usage:

def makeTitle(script, hashtag):
    try:
        genai.configure(api_key='PUT YOUR API KEY')
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content("I am going to give you a script here soon. I want you to take this script which is"
                                          " made for a video and make a video title for the script. Use hashtags, keywords, and make it instesting"
                                          "Please only respond with the title nothing else. I repeat only respond with the tittle. Also don't use emjois"
                                          "I repeat don't use emojis. Make sure it is not NSFW. Please use this hashtag somewhere in the title with a # symbol in front of it: " + hashtag + ". Feel free to use 2-3 more like " + hashtag + " but remember to put a # in front of them. Here is the script: " + script)
        print(response.text)
    except:
        time.sleep(60)
        genai.configure(api_key='PUT YOUR API KEY')
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content("I am going to give you a script here soon. I want you to take this script which is"
                                          " made for a video and make a video title for the script. Use hashtags, keywords, and make it instesting"
                                          "Please only respond with the title nothing else. I repeat only respond with the tittle. Also don't use emjois"
                                          "I repeat don't use emojis. Make sure it is not NSFW. Please use this hashtag somewhere in the title: " + hashtag + ". Here is the script: " + script)
        print(response.text)
    return response.text

def MakeBackgroundSound(script, t):

    try:
        genai.configure(api_key='PUT YOUR API KEY')
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            "Using the script I am about to give you describe a sound for. For example a 128 BPM tech house drum loop."
            " Please only respond with your answer. I repeat nothing else but the music description. Make it less than 20 characters. Here is the script: "
            f"{script}.")

        print(response.text)
    except:
        time.sleep(60)
        genai.configure(api_key='PUT YOUR API KEY')
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            "Using the script I am about to give you describe a sound for. For example a 128 BPM tech house drum loop."
            " Please only respond with your answer. I repeat nothing else but the music description. I repeat nothing else but the music description. Make it less than 20 characters. Here is the script: "
            f"{script}.")

        print(response.text)



    warnings.filterwarnings("ignore", category=FutureWarning)

    # Set the device to GPU if available, otherwise CPU
    device = torch.device("cpu")

    # Download model
    model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    # Move model to the correct device and ensure it uses float32 precision
    model = model.to(device).float()

    # Set up text and timing conditioning
    conditioning = [{
        "prompt": response.text,
        "seconds_start": 0,
        "seconds_total": t
    }]

    # Convert conditioning to float32 if necessary and move it to the right device
    conditioning = [{key: (val if not isinstance(val, torch.Tensor) else val.to(device).float())
                     for key, val in cond.items()} for cond in conditioning]

    # Generate diffusion with valid seed and ensure model is on the correct device
    output = generate_diffusion_cond(
        model,
        steps=100,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=100,  # Reduce from 500 to 100
        sampler_type="dpmpp-3m-sde",
        device=device,
        seed=np.random.randint(0, 2 ** 31 - 1)
    )

    # Rearrange audio batch to a single sequence
    output = rearrange(output, "b d n -> d (b n)")

    # Peak normalize, clip, convert to float32 for saving
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).cpu().numpy()

    gain_factor = 0.2  # Set to 0.5 for half volume or 2.0 for double volume
    output *= gain_factor
    # Ensure correct shape for mono/stereo
    if output.ndim == 1:
        output = output[None, :]  # Reshape to (1, num_samples) for mono

    # Ensure the output is in the correct format for soundfile (float32)
    output = output.astype(np.float32)

    # Save using soundfile (can save as .wav, .flac, etc.)
    sf.write("background.wav", output.T, sample_rate)

que = input("How many vids do you want to make: ")
if que == "post":
    title = makeTitle("Ever wondered why the phases of the moon seem to have a rhythm of their own? Stick around to uncover the fascinating connection between the moon and music. 🌕🎶 Did you know that ancient civilizations used the moon's cycle as a natural way to track time and create musical patterns? It's like a celestial symphony playing out in the night sky. So, the next time you gaze up at the glowing moon, remember that its influence extends beyond the tides and into the realm of music. Stay tuned for more cosmic insights and let the moon's melodies serenade your soul. 🌙🎵✨", "# moonmusic")
    postvid(title)
else:
    que = int(que)
    if(input("do you want to use your own hashtag? t = yes f = no: ").lower() != "t"):
        hashs = getHashtags()
        mainhash = ["", 0]
        for i in range(len(hashs)):
            print(hashs[i])
            try:
                value = checkhashtagvalue(str(hashs[i]))
            except:
                time.sleep(60)
                value = checkhashtagvalue(str(hashs[i]))
            print(value)
            try:
                value = int(value)
                if (value > mainhash[1]):
                    mainhash[0] = hashs[i]
                    mainhash[1] = value
            except:
                value = 0
                if (value > mainhash[1]):
                    mainhash[0] = hashs[i]
                    mainhash[1] = value
        print(mainhash)
    else:
        mainhash = [input("What is the hash? "), 100]
    for video in range(que):
        script = makeScript(mainhash[0])
        sentences = re.split(r'(?<=[.!?])\s+', script)
        clipArray = []
        count = -1
        for sen in sentences:
            count += 1
            makeTTS(sen, count)
            makeImage(sen)
            print(sen)
            try:
                audio = AudioFileClip(f"voiceForTiktok_{count}.wav")
            except OSError as e:
                audio = None
                print(e)
            if(audio != None):
                image_clip = ImageClip("generated_image_Tiktok.png")
                image_clip = image_clip.set_duration(audio.duration)
                screen_size = (1280, 720)
                result = split_string_at_nearest_space(sen)
                result_text = "\n".join(result)
                text_clip = TextClip(result_text, fontsize = 45, color = 'cyan ', stroke_color="black", method='caption', size=screen_size)
                text_clip = text_clip.set_position('center')
                text_clip = text_clip.set_duration(audio.duration)
                text_clip.audio = audio
                video = CompositeVideoClip([image_clip, text_clip])
                clipArray.append(video)
        final_clip = concatenate_videoclips(clipArray)
        MakeBackgroundSound(script=script, t=final_clip.duration)
        background_music = AudioFileClip("background.wav")
        background_music = background_music.set_duration(final_clip.duration)
        combined_audio = CompositeAudioClip([final_clip.audio, background_music])
        final_clip = final_clip.set_audio(combined_audio)
        final_clip.write_videofile("output_video.mp4", codec='libx264', fps=24)
        title = makeTitle(script, mainhash[0])
        postvid(title)
