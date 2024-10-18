#https://ads.tiktok.com/business/creativecenter/inspiration/popular/hashtag/pc/en
import random

from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import numpy as np
import soundfile as sf
import upload_video
import cv2
from pydub import AudioSegment
import warnings
import openai
import json
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
import assemblyai as aai
import google.generativeai as genai
from openai import OpenAI
from moviepy.editor import *
from moviepy.editor import ImageClip, VideoFileClip, concatenate_videoclips
import re
import torch
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, concatenate_videoclips
from huggingface_hub import login
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time
from tiktok_uploader.browsers import get_browser
from tiktok_uploader.auth import AuthBackend
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
    driver.get('https://www.tiktok.com/tiktokstudio/upload?from=upload')

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

    lookingforspace = False
    count = 0
    newTitle = ""
    for letter in description + " ":
        if lookingforspace == False:
            if letter == "#":
                lookingforspace = True

        if lookingforspace:
            if letter == " ":
                lookingforspace = False
                newTitle += f"{Keys.TAB}"

        newTitle += letter


        count+= 1

    print(newTitle)


    caption_input = driver.find_element(By.XPATH, '//div[@contenteditable="true"]')
    caption_input.click()
    for _ in range(25):  # Adjust the range for how many characters you want to delete
        caption_input.send_keys(Keys.BACKSPACE)
    for letter in newTitle:
        if letter == f"{Keys.TAB}":
            time.sleep(3)
        caption_input.send_keys(letter)

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
    genai.configure(api_key='YOUR_API_KEY')
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
    openai.api_key = 'YOUR_API_KEY'

    # Define the function schema
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information that you don't know or need more info on. For example: a game that you don't know about, a move, or some news.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What do you want to search?"
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False
                }
            }
        }
    ]

    # Define your Python function
    def searchWeb(query):
        # This could call a real weather API
        import pprint

        my_api_key = "YOUR_API_KEY"
        my_cse_id = "42c29767c44894cd3"

        def google_search(search_term, api_key, cse_id, amount):
            service = build("customsearch", "v1", developerKey=api_key)
            res = service.cse().list(q=search_term, cx=cse_id, num=amount).execute()
            service = build("customsearch", "v1", developerKey=api_key)
            res2 = service.cse().list(q=search_term, cx=cse_id, num=amount).execute()
            service = build("customsearch", "v1", developerKey=api_key)
            res3 = service.cse().list(q=search_term, cx=cse_id, num=amount).execute()
            service = build("customsearch", "v1", developerKey=api_key)
            res4 = service.cse().list(q=search_term, cx=cse_id, num=amount).execute()
            service = build("customsearch", "v1", developerKey=api_key)
            res5 = service.cse().list(q=search_term, cx=cse_id, num=amount).execute()
            return res['items'] + res2['items'] + res3['items'] + res4['items'] + res5['items']

        results = google_search(
            query, my_api_key, my_cse_id, 10)
        formatted_results = []
        for result in results:
            formatted_results.append({"title": result["title"], "snippet": result["snippet"], "url": result["link"]})

        print(formatted_results)
        return formatted_results

    # Generate a response with function call
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a expert tiktok script writer. You only give scripts that will be read out loud. This means not talking about the scene or what should be displayed. You also dont make future promises. The script you make will have all of the information you want to talk about. It will not talk about diving into a future video."},
            {
                "role": "user",
                "content": "Please write a TikTok script based on the theme I’ll provide. The script should be 60 to 90 seconds long. starting with a strong hook that reappears later for continuity. It must be engaging, concise, and suitable for TikTok. Avoid false claims or personal anecdotes. Aim for informative content, and provide only the spoken script without any additional comments or directions. The content should be appropriate for all audiences and align with TikTok’s guidelines. Please do not talk about the scene. Only talk about the script. You also will not put # at the end. Lastly this is a stand alone video meaning the video should not talk about other videos. If you don't know what it is please search it. If you do search it please still make a script on it using the past criteria. For example the script should still be 60-90 seconds long. Here is the script: " + hashtag,
            }
        ],
        tools=tools,  # Let GPT decide when to call the function
    )
    # Check if GPT called the function
    if response.choices[0].finish_reason == "tool_calls":
        print(response.choices[0].message)
        webfound = searchWeb(eval(response.choices[0].message.tool_calls[0].function.arguments)["query"])
        function_call_result_message = {
            "role": "tool",
            "content": json.dumps({
                "query": webfound
            }),
            "tool_call_id": response.choices[0].message.tool_calls[0].id
        }
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a expert tiktok script writer. You only give scripts that will be read out loud. This means not talking about the scene or what should be displayed. You also dont make future promises. The script you make will have all of the information you want to talk about. It will not talk about diving into a future video."},
                {
                    "role": "user",
                    "content": "Please write a TikTok script based on the theme I’ll provide. The script should be 60 to 90 seconds long. starting with a strong hook that reappears later for continuity. It must be engaging, concise, and suitable for TikTok. Avoid false claims or personal anecdotes. Aim for informative content, and provide only the spoken script without any additional comments or directions. The content should be appropriate for all audiences and align with TikTok’s guidelines. Please do not talk about the scene. Only talk about the script. You also will not put # at the end. Lastly this is a stand alone video meaning the video should not talk about other videos. If you don't know what it is please search it. If you do search it please still make a script on it using the past criteria. For example the script should still be 60-90 seconds long. If you search it, the search script should be about a search result of your choice (you could even combine multible). Here is the script: " + hashtag,
                },
                response.choices[0].message,
                function_call_result_message,
            ]
        )

    # Output the final response
    print(response.choices[0].message)
    return response.choices[0].message.content

def makeImage(prompt, i):
    try:
        genai.configure(api_key='YOUR_API_KEY')
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
        genai.configure(api_key='YOUR_API_KEY')
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
    url = "https://api.getimg.ai/v1/stable-diffusion-xl/text-to-image"

    payload = {
        "model": "real-cartoon-xl-v6",
        "width": 768,
        "height": 768,
        "steps": 90,
        "output_format": "png",
        "response_format": "url",
        "prompt": response.text,
        "negative_prompt": "Disfigured, blurry, nude"
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": "Bearer YOUR_API_KEY"
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
        with open(f"generated_image_Tiktok_{i}.png", "wb") as file:
            file.write(response.content)
        print("Image downloaded successfully!")
    else:
        print(f"Failed to download the image. Status code: {response.status_code}")

def makeTTS(text, i):
    client = OpenAI(
        # This is the default and can be omitted
        api_key="YOUR_API_KEY"
    )

    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,

    )

    with open(f"voiceForTiktok_{i}.wav", "wb") as f:
        f.write(response.content)

    audio = AudioSegment.from_file(f"voiceForTiktok_{i}.wav")

    audio = audio + 5

    # Export the amplified audio back to the file
    audio.export(f"voiceForTiktok_{i}.wav", format="wav")
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
        genai.configure(api_key='YOUR_API_KEY')
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content("I am going to give you a script here soon. I want you to take this script which is"
                                          " made for a video and make a video title for the script. Use hashtags, keywords, and make it instesting"
                                          "Please only respond with the title nothing else. I repeat only respond with the tittle. Also don't use emjois"
                                          "I repeat don't use emojis. Make sure it is not NSFW. Please use this hashtag somewhere in the title with a # symbol in front of it: " + hashtag + ". Feel free to use 2-3 more like " + hashtag + " but remember to put a # in front of them. Here is the script: " + script)
        print(response.text)
    except:
        time.sleep(60)
        genai.configure(api_key='YOUR_API_KEY')
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content("I am going to give you a script here soon. I want you to take this script which is"
                                          " made for a video and make a video title for the script. Use hashtags, keywords, and make it instesting"
                                          "Please only respond with the title nothing else. I repeat only respond with the tittle. Also don't use emjois"
                                          "I repeat don't use emojis. Make sure it is not NSFW. Please use this hashtag somewhere in the title: " + hashtag + ". Here is the script: " + script)
        print(response.text)
    return response.text

def MakeBackgroundSound(script, t):

    try:
        genai.configure(api_key='YOUR_API_KEY')
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            "Using the script I am about to give you describe a sound for. For example a 128 BPM tech house drum loop."
            " Please only respond with your answer. I repeat nothing else but the music description. Make it less than 20 characters. Here is the script: "
            f"{script}.")

        print(response.text)
    except:
        time.sleep(60)
        genai.configure(api_key='YOUR_API_KEY')
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            "Using the script I am about to give you describe a sound for. For example a 128 BPM tech house drum loop."
            " Please only respond with your answer. I repeat nothing else but the music description. I repeat nothing else but the music description. Make it less than 20 characters. . Here is the script: "
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
        "prompt": "128 BPM tech house drum loop",
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

    # Ensure correct shape for mono/stereo
    if output.ndim == 1:
        output = output[None, :]  # Reshape to (1, num_samples) for mono

    # Ensure the output is in the correct format for soundfile (float32)
    output = output.astype(np.float32)

    # Save using soundfile (can save as .wav, .flac, etc.)
    sf.write("output.wav", output.T, sample_rate)

    audio = AudioSegment.from_file(f"output.wav")

    audio = audio - 4

    # Export the amplified audio back to the file
    audio.export(f"output.wav", format="wav")
    print(f"output.wav")

def giggle_zoom_effect(image_path, output_path, duration, fps=24, amplitude=5, zoom_range=(1.0, 1.3)):
    # Load the image as a clip
    clip = ImageClip(image_path).set_duration(duration)

    # Define a function that adds random translations and zoom (for the giggle effect)
    def apply_giggle_zoom(get_frame, t):
        # Get the original image frame
        frame = get_frame(t)
        h, w, _ = frame.shape

        # Create random translations
        tx = amplitude * np.sin(2 * np.pi * t * 2)  # Horizontal wobble
        ty = amplitude * np.cos(2 * np.pi * t * 2)  # Vertical wobble

        # Determine the zoom factor
        zoom_factor = np.interp(t, [0, duration], zoom_range)  # Zoom in from zoom_range[0] to zoom_range[1]

        # Resize the frame for zoom effect
        frame = cv2.resize(frame, (int(w * zoom_factor), int(h * zoom_factor)))

        # Calculate the new position to keep the image centered after zooming
        new_h, new_w, _ = frame.shape
        x_offset = (new_w - w) // 2
        y_offset = (new_h - h) // 2

        # Create transformation matrix for translation
        translation_matrix = np.float32([[1, 0, tx - x_offset], [0, 1, ty - y_offset]])

        # Apply the translation after zoom
        frame = cv2.warpAffine(frame, translation_matrix, (w, h))

        return frame

    # Apply the giggle and zoom effect
    giggle_zoom_clip = clip.fl(apply_giggle_zoom)

    # Write the result as a video or gif
    giggle_zoom_clip.write_videofile(output_path, fps=fps)


# Example usage:


que = input("How many vids do you want to make: ")
if que == "youtube":
    title = "Dashain: Celebrating Victory & Togetherness | #HappyDashain #NepaliCulture #FestivalVibes"
    upload_video.upload_Video_youtube(title, title)
elif que == "post":
    title = makeTitle("Get ready to hit a grand slam at the Los Angeles Dogers. Whether you're a die hard fan or just getting into baseball, the Dodgers have a long history of success and excitement waiting for. From Epic games to fan favorites, the Dodgers bring the heat on and off the field. From Epic games to fan favorites, the Dodgers bring the heat on and off the field. Follow the official Dodgers accounts on Twitter, Instagram and more to be part of the action. Don't miss out on being part of the Dodger Blue family.", "#Dodger ")
    cleaned_text = title.replace("*", "")
    postvid(cleaned_text)
    upload_video.upload_Video_youtube(cleaned_text, cleaned_text)
elif que == "c":
    mainhash = ["#5lifehacks", 0]
    script = "Introducing five incredible life hacks that will revolutionize your daily routine. Stay tuned for some mind-blowing tips! Let's dive in:\n\nLife Hack #1: Discover a genius trick to detangle your headphones effortlessly.\nLife Hack #2: Learn how to remove stubborn sticker residue like a pro.\nLife Hack #3: Find out a quick way to check if your sunglasses are polarized.\nLife Hack #4: Say goodbye to water stains in wood using a surprising household item.\nLife Hack #5: Transform hard butter into baking-ready softness in seconds with a simple tool.\n\nGet ready to upgrade your life with these game-changing hacks. Try them out and thank me later!"
    sentences = re.split(r'(?<=[.!?])\s+', script)
    clipArray = []
    count = -1
    for sen in sentences:
        count += 1
        result = split_string_at_nearest_space(sen)
        try:
            audio = AudioFileClip(f"voiceForTiktok_{count}.wav")
        except:
            makeTTS(sen, count)
        try:
            audio = AudioFileClip(f"voiceForTiktok_{count}.wav")
        except OSError as e:
            audio = None
            print(e)

        try:
            vid_clip = VideoFileClip(f"generated_image_Tiktok_{count}_giggle.mp4")
        except:
            makeImage(sen, count)
            giggle_zoom_effect(
                rf"C:\Users\jgdiv\PycharmProjects\pythonProject\tiktokAutoPoster\generated_image_Tiktok_{count}.png",
                rf"generated_image_Tiktok_{count}_giggle.mp4", duration=audio.duration)
        print(sen)
        if audio != None:
            vid_clip = VideoFileClip(f"generated_image_Tiktok_{count}_giggle.mp4")
            screen_size = (1280, 720)
            result_text = "\n".join(result)

            aai.settings.api_key = "YOUR_API_KEY"
            transcriber = aai.Transcriber()

            transcript = transcriber.transcribe(f"voiceForTiktok_{count}.wav")
            # transcript = transcriber.transcribe("./my-local-audio-file.wav")

            print(transcript.words)
            text_clips = []
            for word in transcript.words:
                text_clip = TextClip(word.text, fontsize=100, color='cyan ', stroke_color="black", method='caption',
                                     size=screen_size)
                text_clip = text_clip.set_position('center')
                text_clip = text_clip.set_start(word.start / 1000)
                text_clip = text_clip.set_duration(word.end / 1000 - word.start / 1000)
                text_clip.audio = audio
                text_clips.append(text_clip)
            video = CompositeVideoClip([vid_clip] + text_clips)
            clipArray.append(video)
    video_clips = []

    first = True
    # Loop through each video path and load the video
    for video in clipArray:

        # If it's not the first video, apply crossfade transition
        if first:
            first = False
        else:
            video = video.crossfadein(1.5)

        # Append the processed video to the list
        video_clips.append(video)
    final_clip = concatenate_videoclips(video_clips, method="compose")
    try:
        background_music = AudioFileClip("output.wav")
        background_music = background_music.set_duration(final_clip.duration)
    except:
        MakeBackgroundSound(script=script, t=final_clip.duration)
        background_music = AudioFileClip("output.wav")
        background_music = background_music.set_duration(final_clip.duration)
    combined_audio = CompositeAudioClip([final_clip.audio, background_music])
    final_clip = final_clip.set_audio(combined_audio)
    final_clip.write_videofile("output_video.mp4", codec='libx264', fps=24)
    title = makeTitle(script, mainhash[0])
    cleaned_text = title.replace("*", "")
    postvid(cleaned_text)
    upload_video.upload_Video_youtube(cleaned_text, cleaned_text)
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
                if (value >= mainhash[1]):
                    if random.randint(1, 2) == 1:
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
            try:
                audio = AudioFileClip(f"voiceForTiktok_{count}.wav")
            except OSError as e:
                audio = None
                print(e)
            makeImage(sen, count)
            giggle_zoom_effect(
                rf"C:\Users\jgdiv\PycharmProjects\pythonProject\tiktokAutoPoster\generated_image_Tiktok_{count}.png",
                rf"generated_image_Tiktok_{count}_giggle.mp4", duration=audio.duration)
            print(sen)
            if audio != None:
                vid_clip = VideoFileClip(f"generated_image_Tiktok_{count}_giggle.mp4")
                screen_size = (1280, 720)
                result = split_string_at_nearest_space(sen)
                result_text = "\n".join(result)

                aai.settings.api_key = "YOUR_API_KEY"
                transcriber = aai.Transcriber()

                transcript = transcriber.transcribe(f"voiceForTiktok_{count}.wav")
                text_clips = []
                for word in transcript.words:
                    text_clip = TextClip(word.text, fontsize=100, color='cyan ', stroke_color="black", method='caption',
                                         size=screen_size)
                    text_clip = text_clip.set_position('center')
                    text_clip = text_clip.set_start(word.start / 1000)
                    text_clip = text_clip.set_duration(word.end / 1000 - word.start / 1000)
                    text_clip.audio = audio
                    text_clips.append(text_clip)
                video = CompositeVideoClip([vid_clip] + text_clips)
                clipArray.append(video)

        video_clips = []

        first = True
        # Loop through each video path and load the video
        for video in clipArray:

            # If it's not the first video, apply crossfade transition
            if first:
                first = False
            else:
                video = video.crossfadein(1.5)

            # Append the processed video to the list
            video_clips.append(video)
        final_clip = concatenate_videoclips(video_clips, method="compose")
        MakeBackgroundSound(script=script, t=final_clip.duration)
        background_music = AudioFileClip("output.wav")
        background_music = background_music.set_duration(final_clip.duration)
        combined_audio = CompositeAudioClip([final_clip.audio, background_music])
        final_clip = final_clip.set_audio(combined_audio)
        final_clip.write_videofile("output_video.mp4", codec='libx264', fps=24)
        title = makeTitle(script, mainhash[0])
        cleaned_text = title.replace("*", "")
        postvid(cleaned_text)
        upload_video.upload_Video_youtube(cleaned_text, cleaned_text)
