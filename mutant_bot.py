import discord
from discord.ext import commands
import os
import wave
import asyncio
from discord.ext import voice_recv
import time
from datetime import datetime
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
import shutil
from threading import Timer, Lock
from dotenv import load_dotenv
import numpy as np
import whisper

# Replace 'YOUR_BOT_TOKEN' with your actual bot token
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
HQ_TRANSCRIPTION = os.getenv("HQ_TRANSCRIPTION") == "true"
HQ_MODEL_SIZE = os.getenv("HQ_MODEL_SIZE")

# ========================= CONFIGURABLE PARAMETERS =========================

# Time interval in seconds for splitting audio into new recording files
RECORDING_SPLIT_INTERVAL = 30  # Split recording every 30 seconds

# Time threshold in seconds for detecting silence to finalize a recording
SILENCE_THRESHOLD = 10  # Finalize recording if no activity for 10 seconds

# Time interval in seconds for checking new finalized recordings for transcription
TRANSCRIPTION_CHECK_INTERVAL = 5  # Check for new files every 5 seconds

# Time interval in seconds for muting a user after offensive speech is detected
MUTE_DURATION = 15  # Mute user for 15 seconds after offensive statement

# Setting up intents to manage permissions for the bot
intents = discord.Intents.default()
intents.voice_states = True
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)
recording_clients = {}  # A dictionary to store recording states per user
transcription_tasks = {}  # A dictionary to manage transcription tasks per guild

model = None
if HQ_TRANSCRIPTION:
    # Load the Whisper model (e.g., "base", "small", "medium", "large")
    model = whisper.load_model(HQ_MODEL_SIZE)
else:
    # Load the pre-trained German model and processor for transcription
    model_name = "facebook/wav2vec2-large-xlsr-53-german"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

# Load the German hate speech detection model for sentiment analysis
device_index = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "text-classification",
    model="deepset/bert-base-german-cased-hatespeech-GermEval18Coarse",
    device=device_index
)

# Clean up recordings folder on startup
if os.path.exists('recordings'):
    shutil.rmtree('recordings')
os.makedirs('recordings', exist_ok=True)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name} - {bot.user.id}')

@bot.command()
async def join(ctx):
    if ctx.author.voice:
        channel = ctx.author.voice.channel
        if ctx.voice_client is None:
            await channel.connect(cls=voice_recv.VoiceRecvClient)
            await ctx.send(f'Joined {channel}')
        else:
            await ctx.send('I am already connected to a voice channel.')
    else:
        await ctx.send('You are not connected to a voice channel.')

@bot.command()
async def leave(ctx):
    if ctx.voice_client:
        await ctx.voice_client.disconnect()
        await ctx.send('Disconnected from the voice channel.')
    else:
        await ctx.send('I am not connected to any voice channel.')

@bot.command()
async def record(ctx):
    if not ctx.voice_client:
        if ctx.author.voice:
            # If not connected, connect to the user's channel and start recording
            channel = ctx.author.voice.channel
            await channel.connect(cls=voice_recv.VoiceRecvClient)
            await ctx.send(f'Joined {channel} and started recording.')
        else:
            await ctx.send('You are not connected to a voice channel.')
            return
    else:
        voice_client = ctx.voice_client

    # Get list of members in the voice channel, excluding bots
    channel = ctx.voice_client.channel
    members = [member for member in channel.members if not member.bot]

    if not members:
        await ctx.send('No members found to record in the voice channel.')
        return
    
    # Start listening to the voice channel and record members
    receiver = MyVoiceReceiver(members)
    ctx.voice_client.listen(receiver)
    recording_clients[ctx.guild.id] = receiver
    await ctx.send('Recording started for all members. Type !stop to end.')

    # Start the transcription task
    transcription_task = asyncio.create_task(transcribe_recordings(ctx))
    transcription_tasks[ctx.guild.id] = transcription_task

# Custom audio receiver class for recording voice channel members, specifically tailored to create small audio packages for each user (contains a lot of edge case handling)
class MyVoiceReceiver(voice_recv.AudioSink):
    def __init__(self, members_to_record):
        super().__init__()
        self.members_to_record = {member.name for member in members_to_record}  # Store the names of members to record
        self.files = {}  # Dictionary to manage files per user
        self.start_times = {}  # Dictionary to track start times per user
        self.silence_timers = {}  # Dictionary to track silence timers per user
        self.silence_threshold = SILENCE_THRESHOLD  # Silence threshold in seconds
        self.lock = Lock()  # Lock to prevent race conditions

        for member in members_to_record:
            # Create a directory for each user if it doesn't exist
            user_dir = f'recordings/{member.name}'
            os.makedirs(user_dir, exist_ok=True)
            finalized_dir = f'recordings/{member.name}/finalized'
            os.makedirs(finalized_dir, exist_ok=True)
            # Initialize start time for snippet tracking
            self.start_times[member.name] = time.time()
            # Create initial wave file for each user
            self._create_new_wav_file(member.name)
            # Initialize silence timer for each user
            self._reset_silence_timer(member.name)

    def _create_new_wav_file(self, user_name):
        # Create a new wave file for the user with a timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = f'recordings/{user_name}/audio_{user_name}_{timestamp}.wav'
        wav_file = wave.open(file_path, 'wb')
        wav_file.setnchannels(2)  # Set the number of audio channels (stereo)
        wav_file.setsampwidth(2)  # Set sample width to 16-bit audio
        wav_file.setframerate(48000)  # Set the audio sample rate
        self.files[user_name] = {'wav_file': wav_file, 'file_path': file_path, 'is_closed': False}
        # Start the silence timer for the new file
        self._reset_silence_timer(user_name)

    def wants_opus(self) -> bool:
        return False  # Indicate that we want raw PCM data, not Opus

    def write(self, user, data):
        if data.source is not None:
            # Write PCM data to the wave file for the user if they are being recorded
            if user.name in self.members_to_record:
                with self.lock:
                    current_time = time.time()
                    # Check if RECORDING_SPLIT_INTERVAL seconds have passed to create a new file
                    if current_time - self.start_times[user.name] >= RECORDING_SPLIT_INTERVAL:
                        # Close the current file and move it to the finalized folder
                        self._finalize_recording(user_name=user.name, lock_acquired=True)
                        # Create a new wave file
                        self._create_new_wav_file(user_name=user.name)
                        # Reset the start time
                        self.start_times[user.name] = current_time
                    # Write the PCM data to the current file if it is still open
                    if not self.files[user.name]['is_closed']:
                        self.files[user.name]['wav_file'].writeframes(data.pcm)
                    # Reset the silence timer since user is speaking
                    self._reset_silence_timer(user.name)

    def _finalize_recording(self, user_name, lock_acquired=False):
        # Close the current file and move it to the finalized folder
        if not lock_acquired:
            self.lock.acquire()
        try:
            if user_name in self.files and not self.files[user_name]['is_closed']:
                file_path = self.files[user_name]['file_path']
                if os.path.exists(file_path):
                    self.files[user_name]['wav_file'].close()
                    self.files[user_name]['is_closed'] = True
                    finalized_dir = f'recordings/{user_name}/finalized'
                    try:
                        shutil.move(file_path, finalized_dir)
                        print(f'Finalized recording for user: {user_name}')
                    except FileNotFoundError:
                        print(f'File not found when trying to finalize recording for user: {user_name}')
        finally:
            if not lock_acquired:
                self.lock.release()

    def _reset_silence_timer(self, user_name):
        # Cancel the existing timer if it exists
        if user_name in self.silence_timers:
            self.silence_timers[user_name].cancel()
        # Set a new timer to finalize the recording after silence threshold
        self.silence_timers[user_name] = Timer(self.silence_threshold, self._handle_silence_timeout, [user_name])
        self.silence_timers[user_name].start()

    def _handle_silence_timeout(self, user_name):
        # Handle the timeout caused by silence
        self._finalize_recording(user_name=user_name)
        print(f'Silence detected. Finalized recording for user: {user_name}')

    def cleanup(self):
        # Close all wave files after recording ends and move them to the finalized folder
        with self.lock:
            for user_name, file_data in self.files.items():
                if not file_data['is_closed']:
                    file_data['wav_file'].close()
                    file_data['is_closed'] = True
                    finalized_dir = f'recordings/{user_name}/finalized'
                    if os.path.exists(file_data['file_path']):
                        try:
                            shutil.move(file_data['file_path'], finalized_dir)
                            print(f'Saved recording for user with name {user_name}')
                        except FileNotFoundError:
                            print(f'File not found when trying to save recording for user: {user_name}')
            # Cancel all silence timers
            for timer in self.silence_timers.values():
                timer.cancel()

@bot.command()
async def stop(ctx):
    if ctx.voice_client:
        voice_client = ctx.voice_client
        if ctx.guild.id in recording_clients:
            receiver = recording_clients.pop(ctx.guild.id)
            voice_client.stop_listening()
            # Clean up and save recorded files if the listener is MyVoiceReceiver
            receiver.cleanup()
            await ctx.send('Stopped recording and saved for all users.')
        await ctx.voice_client.disconnect()

        # Cancel the transcription task if it exists
        if ctx.guild.id in transcription_tasks:
            transcription_task = transcription_tasks.pop(ctx.guild.id)
            transcription_task.cancel()
            await ctx.send('Transcription task stopped.')
    else:
        await ctx.send('I am not connected to any voice channel.')

# Subroutine to mute and unmute a user asynchronously
async def mute_user(guild, user_name):
    member = discord.utils.get(guild.members, name=user_name)
    if member:
        try:
            await member.edit(mute=True)
            await asyncio.sleep(MUTE_DURATION)
            await member.edit(mute=False)
            await guild.system_channel.send(f'**{user_name}** has been unmuted.')
        except discord.HTTPException as e:
            print(f"Failed to mute/unmute user {user_name}: {e}")

def transcribe_wav2vec(y):
    # Convert audio to tensor and normalize
    input_values = processor(y, return_tensors="pt", sampling_rate=16000).input_values.to(device)
    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits
    # Decode logits to transcription
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

def transcribe_whisper(y):
    audio = y / np.max(np.abs(y))
    result = model.transcribe(audio, language="de")
    return result['text']

# Function to transcribe recordings periodically
async def transcribe_recordings(ctx):
    while True:
        await asyncio.sleep(TRANSCRIPTION_CHECK_INTERVAL)  # Wait for TRANSCRIPTION_CHECK_INTERVAL seconds between checks
        for user_name in os.listdir('recordings'):
            user_dir = f'recordings/{user_name}/finalized'
            for file_name in os.listdir(user_dir):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(user_dir, file_name)
                    # Load and resample audio with librosa
                    y, sr = librosa.load(file_path, sr=16000)  # The model expects 16kHz audio

                    duration = len(y) / sr
                    if duration < 1.0:
                        print(f"Ignoring {file_name} as it is less than 1 second.")
                        continue  # Skip processing for short files

                    if len(y) != 0:
                        transcription = ''
                        if HQ_TRANSCRIPTION:
                            transcription = transcribe_whisper(y)
                        else:
                            transcription = transcribe_wav2vec(y)

                        # Perform sentiment analysis
                        sentiment_result = classifier(transcription)[0]
                        is_offensive = 'Offensive' if sentiment_result['label'] == 'OFFENSE' else 'Non-offensive'
                        # Send the transcription to the Discord channel with sentiment analysis
                        await ctx.send(f'**{user_name} said:** {transcription} - **Sentiment:** {is_offensive}')
                    
                        # If the message is offensive, mute the user for MUTE_DURATION seconds
                        if sentiment_result['label'] == 'OFFENSE':
                            await ctx.send(f'**{user_name}** has been muted for {MUTE_DURATION} seconds due to offensive speech.')
                            asyncio.create_task(mute_user(ctx.guild, user_name))

                    # Remove the file after processing
                    os.remove(file_path)


bot.run(TOKEN)