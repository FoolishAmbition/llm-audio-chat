# llm-audio-chat
Real-Time LLM Voice Chat with Python, FastRTC, and Open Source AI

## Repository layout

| File | Purpose |
| --- | --- |
| `folders/01.py` | Microphone loopback demo. |
| `folders/02.py` | Voice chat with Moonshine transcription, Groq replies, and Kokoro playback. |
| `folders/03.py` | Voice chat with Groq Whisper transcription, Groq replies, and Kokoro playback. |
| `folders/test.py` | FastRTC version check. |
| `context/` | Local planning documents, including the PRD and milestone tracker. Git ignores this directory. |

Features
- 🎙️ Real-time voice input using microphone
- 🧠 Speech-to-text powered by **Groq + OpenAI Whisper**
- 💬 LLM response generation (Groq)
- 🔊 AI voice synthesis with local **Kokoro**
- ⚡ Powered by **FastRTC** for ultra-low-latency audio streaming
