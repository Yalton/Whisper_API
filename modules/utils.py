import yt_dlp

from fastapi import Header, HTTPException, Security
from .config import settings

def get_auth_token(authorization: str = Header(...)):
    """
    Dependency to extract and validate the authorization token from the request headers.
    """
    if authorization != settings.AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing token")
    return authorization


def download_youtube_audio(youtube_url: str, download_path: str) -> str:
    """
    Downloads the audio from a YouTube URL and saves it to the specified download path.
    
    Args:
        youtube_url (str): The URL of the YouTube video to download.
        download_path (str): The directory where the audio file should be saved.
    
    Returns:
        str: The path to the downloaded audio file.
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': download_path + '/%(id)s.%(ext)s',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        ydl.download([youtube_url])
        filename = ydl.prepare_filename(info_dict)
        # Update the filename to the correct format as per the postprocessor
        audio_file_path = download_path + '/' + info_dict['id'] + '.mp3'
    
    return audio_file_path
