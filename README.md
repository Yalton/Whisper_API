<a name="readme-top"></a>

[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Yalton/Whisper_API">
    <!-- You can add a logo here -->
  </a>

<h3 align="center">Local ASR API</h3>

</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

This project provides an easy-to-use, self-hosted speech-to-text (ASR) API for
audio files and YouTube videos. It serves two local backends side by side, and
each request picks one:

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper), the mature and
  resource-efficient default.
- [NVIDIA Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), a fast
  FastConformer-TDT model for English and 24 other European languages.

This is an ASR/speech-to-text service. It does not perform TTS/text-to-speech.

### Features:

- Audio file transcription
- YouTube video audio transcription
- Language detection
- Whisper or Parakeet inference, chosen per request
- Authorization for secure API access

### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples:
- [FastAPI](https://fastapi.tiangolo.com/)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper.git)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) (Parakeet)


## Getting Started

To get a local copy up and running follow these simple steps. This project supports both a traditional Python environment setup and Docker for containerization, catering to various deployment preferences.

### Prerequisites

- Python 3.9 or higher (Python 3.12 is recommended)
- Docker and Docker Compose (for running with Docker)
- NVIDIA Docker Runtime (if using CUDA for GPU acceleration)

### Installation

#### Running Locally Without Docker

1. Clone the repo:
   ```sh
   git clone https://github.com/Yalton/Whisper_API.git
   ```
2. Navigate to the project directory:
   ```sh
   cd Whisper_API
   ```
3. Install required packages:
   ```sh
   pip install -r requirements.txt
   ```
4. Run it:
   ```sh
   python3 main.py
   ```
#### Running With Docker (Recommened)

1. Clone the repo if you haven't already done so:
   ```sh
   git clone https://github.com/Yalton/Whisper_API.git
   ```
2. Navigate to the project directory:
   ```sh
   cd Whisper_API
   ```
3. Build the Docker container:
   ```sh
   docker compose build
   ```
4. Run the container:
   ```sh
   docker compose up -d
   ```

### Usage 

Getting a basic transcription is easy, simply invoke the following request 

```sh
curl http://localhost:8000/v1/audio/transcriptions/ \
  -H 'Authorization: Bearer auth_token' \
  -H "Content-Type: multipart/form-data" \
  -F 'file=@PickleRick-MasterChief.mp3;type=video/webm'
```

You will then receive a json respone resembling the following 

```json
{
    "text": " This is Steve Downs, the voice of Master Chief Sierra 117.  Sir, I'm here to report that there's this guy named Rick.  And I kid you not, he turns himself into a pickle.  He's called Pickle Rick.  Funniest shit I've ever seen.  Glory to the GDI.  Chief, out.",
    "task": "transcribe",
    "language": "en",
    "duration": 23.289625,
    "language_probability": 0.9990234375,
    "filename": "PickleRick-MasterChief.mp3",
    "model": "large-v3"
}
```

This can also be done with [youtube videos](https://www.youtube.com/watch?v=2Hm8MB9Jx1k)

```sh
curl http://localhost:8000/v1/audio/transcriptions/  \
-H 'Authorization: auth_token' \
-H "Content-Type: multipart/form-data"   \
-F 'youtube_url=https://www.youtube.com/watch?v=2Hm8MB9Jx1k'
```

`Response` 

```json
{
    "text": " I live in a low-income housing environment that goes by the government name of Section 8.  Me and a group of my allies control certain areas of this section in order to run our illegitimate business.  We possess unregistered firearms, stolen vehicles, mind-altering inhibitors, and only use cash for financial purchases.  If anyone would like to settle unfinished altercations, I will be more than happy to release my address.  I would like to warn you, I am a very dangerous person.  And I regularly disobey the law.",
    "task": "transcribe",
    "language": "en",
    "duration": 32.357,
    "language_probability": 0.994140625,
    "filename": "https://www.youtube.com/watch?v=2Hm8MB9Jx1k"
}
```

Gathering timestamps for the audio sample is also supported; there are two granularity parameters `segment` and `word`

Segment breaks the clip up into segments and provides timestamps for those segments 

Word breaks the clip up into words and provides timestamps for every detected word

#### Example of segment 

```sh
curl http://localhost:8000/v1/audio/transcriptions/ \
  -H 'Authorization: auth_token' \
  -H "Content-Type: multipart/form-data" \
  -F 'youtube_url=https://www.youtube.com/watch?v=2Hm8MB9Jx1k'
  -F "timestamp_granularities=segment" \
  -F "response_format=verbose_json"
```

The response: [Response](docs/segment.json)

#### Example of word 

```sh
curl http://localhost:8000/v1/audio/transcriptions/ \
  -H 'Authorization: auth_token' \
  -H "Content-Type: multipart/form-data" \
  -F 'youtube_url=https://www.youtube.com/watch?v=2Hm8MB9Jx1k'
  -F "timestamp_granularities=word" \
  -F "response_format=verbose_json"
```
The response: [Response](docs/word.json)

#### Prompt

`prompt` works as in the OpenAI API: Whisper continues in the style of the
text you give it. Without one, large-v3 can transcribe a whole stream in
lowercase with no punctuation, which leaves callers no sentence boundaries.
A short punctuated prompt such as `Hello, everyone. Welcome back!` fixes
that. Parakeet ignores it and punctuates on its own.

#### Choosing a backend

The OpenAI-style `model` field picks the backend for each request:

- `parakeet`, or any name containing it such as `nvidia/parakeet-tdt-0.6b-v3`,
  uses Parakeet.
- `whisper-1`, or a Whisper size such as `large-v3`, uses Whisper.
- Anything else, or no `model` field, uses `ASR_BACKEND` (Whisper by default).

```sh
curl http://localhost:8001/v1/audio/transcriptions \
  -H "Authorization: Bearer ${AUTH_TOKEN}" \
  -F "file=@speech.wav" \
  -F "model=parakeet"
```

The response's `model` names the checkpoint that ran. Each model loads on its
first request and is released after `ASR_IDLE_TIMEOUT` seconds unused,
independently of the other, so both can be resident at once.


### Configuration

#### Environment Variables

The application's behavior can be customized using environment variables. When running with Docker, these can be set in the `docker-compose.yml` file under the `environment` section. Here are the variables you can configure:

- `AUTH_TOKEN`: Token used for authenticating to the API. Supply it as
  `Authorization: Bearer <token>`. The legacy raw-token header remains supported.
- `ASR_BACKEND`: `whisper` (default) or `parakeet`; the backend for requests
  whose `model` field names neither.
- `WHISPER_MODEL_SIZE`: The model size of the Whisper model to use (e.g., `tiny`, `base`, `small`, `medium`, `large`, `large-v2`).
- `COMPUTE_DEVICE`: Set to `cuda` to use GPU acceleration, or `cpu` for CPU processing.
- `WHISPER_BATCH_SIZE`: Whisper only. Above `0`, uses faster-whisper's
  batched pipeline, which splits audio on speech with Silero VAD and decodes
  that many segments per GPU batch. Defaults to `0` (sequential). On an RTX
  4090, an hour of film audio took 24 s instead of 331 s; at batch size 8 the
  process peaked at about 7.5 GB of VRAM versus about 4.5 GB sequential. Batching
  loses speech in fast commentary over video audio: on 180 s of a reaction
  stream it kept 408 of the 608 words the sequential decoder found.
- `PARAKEET_MODEL`: NeMo checkpoint. Defaults to
  `nvidia/parakeet-tdt-0.6b-v3`; `nvidia/parakeet-tdt-0.6b-v2` is English only.
- `PARAKEET_CHUNK_SECONDS`: Parakeet transcribes at most 20 minutes in one
  pass, so it always splits longer audio into chunks of this length, or of
  `ASR_CHUNK_SECONDS` if that is shorter, even when `ASR_CHUNK_SECONDS` is `0`.
  Defaults to `1200`.
- `PARAKEET_LOCAL_ATTENTION`: Parakeet's limited-context attention window, in
  80 ms frames each side. Defaults to `256`, as NVIDIA recommends for long
  audio. `0` restores full attention, which ran out of memory on a 20-minute
  chunk on a 24 GB RTX 4090.
- `ASR_IDLE_TIMEOUT`: Seconds before inactive model weights are released.
  Defaults to `300`; set to `0` to keep loaded models resident.
- `NVIDIA_VISIBLE_DEVICES`: Specifies which GPUs to use (e.g., `all`, `0`, `1`).
- `NVIDIA_DRIVER_CAPABILITIES`: Set to `compute,utility` for CUDA computations.

Here's an example of how the `environment` section might look in your `docker-compose.yml` file:

```yaml
environment:
  - ASR_BACKEND=whisper
  - WHISPER_MODEL_SIZE=large-v3
  - PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v3
  - ASR_IDLE_TIMEOUT=300
  - COMPUTE_DEVICE=cuda
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

Parakeet punctuates and capitalizes on its own and ignores `prompt`. It does
not report the language it heard: `language` echoes the request's `language`
field, or is `null` without one, and `language_probability` and word-level
`probability` are `null`. The language detection endpoint takes the same
`model` field and returns `400` for Parakeet. Chunks are cut at fixed times, so a word spoken across a 20-minute
boundary can be split or lost.

#### Editing Environment Variables

To change the environment variables:

1. Open the `docker-compose.yml` file in a text editor.
2. Locate the `environment` section under the `whisper_api` service.
3. Edit the values as needed.
4. Save the file and restart the Docker container for the changes to take effect:
   ```sh
   docker compose down
   docker compose up -d
   ```

Now, your Whisper API is set up and running, either locally or within a Docker container, based on your setup choice. Follow the next sections for how to use the API and contribute to its development.

The Docker Compose host port defaults to `8001` to avoid colliding with other local APIs;
override it with `WHISPER_API_PORT`. Navigate to http://localhost:8001/docs for Swagger
API documentation when using the default Compose configuration.

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<!-- CONTACT -->
## Contact

Your Name - [drbailey117@gmail.com](mailto:drbailey117@gmail.com)

Project Link: [https://github.com/Yalton/Whisper_API](https://github.com/Yalton/Whisper_API)

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [FastAPI](https://fastapi.tiangolo.com/)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/Yalton/Whisper_API.svg?style=for-the-badge
[contributors-url]: https://github.com/Yalton/Whisper_API/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Yalton/Whisper_API.svg?style=for-the-badge
[forks-url]: https://github.com/Yalton/Whisper_API/network/members
[stars-shield]: https://img.shields.io/github/stars/Yalton/Whisper_API.svg?style=for-the-badge
[stars-url]: https://github.com/Yalton/Whisper_API/stargazers
[issues-shield]: https://img.shields.io/github/issues/Yalton/Whisper_API.svg?style=for-the-badge
[issues-url]: https://github.com/Yalton/Whisper_API/issues
[license-shield]: https://img.shields.io/github/license/Yalton/Whisper_API.svg?style=for-the-badge
[license-url]: https://github.com/Yalton/Whisper_API/blob/master/LICENSE.txt
[product-screenshot]: images/screenshot.png
