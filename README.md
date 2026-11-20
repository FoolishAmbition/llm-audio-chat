# llm-audio-chat
Real-Time LLM Voice Chat with Python, FastRTC, and Open Source AI

## Repository layout

| File | Purpose |
| --- | --- |
| `desktop_assistant.py` | Windows desktop assistant for voice and text. |
| `demos/microphone_loopback.py` | Microphone loopback demo. |
| `demos/local_speech_chat.py` | Local voice chat with Faster Whisper, Ollama, and Kokoro. |
| `demos/groq_whisper_chat.py` | Voice chat with Groq Whisper transcription, Groq replies, and Kokoro playback. |
| `demos/check_fastrtc_version.py` | FastRTC version check. |
| `context/` | Local planning documents, including the PRD and milestone tracker. Git ignores this directory. |

## Run the Windows desktop app

Start Ollama and make sure `qwen2.5:7b` is installed. From the project folder, install the dependencies into the project environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen2.5:7b
```

Double-click **Start Voice Assistant.cmd**. Type a question and press Enter, or click **Enable voice mode** and start speaking. A short pause sends your turn automatically. The app resumes listening after speaking. Click **Disable voice mode** to turn off the microphone. The first voice turn downloads the selected Whisper model and Kokoro if needed. Once cached, both can run offline. Voice and text share the current conversation.

Open **Voice settings** to choose a microphone and Whisper model, adjust pause length and speech sensitivity, or change the voice, speaking speed, and reply length. New settings default to a 2.5-second pause, adjustable up to 5 seconds. For long questions, enable **Wait for me to send**, speak with voice mode on, then click **Send voice turn**. Settings are saved locally between launches. If a saved microphone is unavailable, the app uses the system default. Voice mode starts off on every launch. Chat renders Markdown replies, while voice playback reads the text without Markdown markers.

Under **Words to recognize**, enter names or terms Whisper often mishears, separated by commas. Try the same spoken phrases with the list empty and then filled. The app saves this list locally between launches. To clear it, erase the field.

Use **Set up your voice** to read practice phrases, replay each clip, and correct what Whisper heard. **Approve and save sample** saves a WAV and its exact transcript to the local app data folder shown at the end of setup. Skipping, re-recording, or closing discards unapproved clips. The saved samples are a dataset for later fine-tuning.

Voice setup includes 20 English prompts and resumes with the phrases you have yet to approve. **Voice profile** shows whether saved vocabulary is loaded, how many recordings you approved, and which Whisper model is selected.

## Run the browser demo

Start Ollama and run these commands in PowerShell from the repository root:

```powershell
ollama pull qwen2.5:7b
python -m pip install -r requirements.txt
python demos\local_speech_chat.py
```

The first run downloads Faster Whisper `medium` and Kokoro. Open the local URL and allow microphone access. Speak, pause, and listen for the reply. On later runs, set `$env:HF_HUB_OFFLINE = '1'` before launching to use the cached models without contacting Hugging Face.

To try a different transcription model, restart with `python demos\local_speech_chat.py --whisper-model small`. Download each model once before using it offline.

Features
- 🎙️ Real-time voice input using microphone
- 🧠 Speech-to-text powered by **Groq + OpenAI Whisper**
- 💬 LLM response generation (Groq)
- 🔊 AI voice synthesis with local **Kokoro**
- ⚡ Powered by **FastRTC** for ultra-low-latency audio streaming
