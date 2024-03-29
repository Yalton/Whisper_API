<a name="readme-top"></a>

[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Yalton/Whisper_API">
    <!-- You can add a logo here -->
  </a>

<h3 align="center">Whisper API</h3>

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

This project aims to provide an easy to use self hostable solution for transcribing audio files either from local files or youtube videos. It uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper.git) which is version of OpenAIs whisper model that is up to 4x as fast as the original model. 

This project aimed to place this model behind an easy to use API such that it can be consumed by other projects that require quick transcription. 

### Features:

- Audio file transcription
- YouTube video audio transcription
- Language detection
- Authorization for secure API access

### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples:
- [FastAPI](https://fastapi.tiangolo.com/)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper.git)


## Getting Started

To get a local copy up and running follow these simple steps. This project supports both a traditional Python environment setup and Docker for containerization, catering to various deployment preferences.

### Prerequisites

- Python 3.8 or higher (for running without Docker)
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
  -H 'Authorization: auth_token' \
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
    "filename": "PickleRick-MasterChief.mp3"
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


### Configuration

#### Environment Variables

The application's behavior can be customized using environment variables. When running with Docker, these can be set in the `docker-compose.yml` file under the `environment` section. Here are the variables you can configure:

- `AUTH_TOKEN`: Token used for Authenticating to the API, make it whatever you want 
- `WHISPER_MODEL_SIZE`: The model size of the Whisper model to use (e.g., `tiny`, `base`, `small`, `medium`, `large`, `large-v2`).
- `COMPUTE_DEVICE`: Set to `cuda` to use GPU acceleration, or `cpu` for CPU processing.
- `NVIDIA_VISIBLE_DEVICES`: Specifies which GPUs to use (e.g., `all`, `0`, `1`).
- `NVIDIA_DRIVER_CAPABILITIES`: Set to `compute,utility` for CUDA computations.

Here's an example of how the `environment` section might look in your `docker-compose.yml` file:

```yaml
environment:
  - WHISPER_MODEL_SIZE=large-v3
  - COMPUTE_DEVICE=cuda
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

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

Navigate to http://localhost:8000/docs for the Swagger API Docs and testing the API 

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
