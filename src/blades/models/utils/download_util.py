# taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
import os

import requests


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    
    response = session.get(URL, params={'_id': id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        params = {'_id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == "__main__":
    file_id = "1rdRFbKeT9woS48Fmmo2mgJWDWSexhGeS"
    destination = './femnist.zip'
    download_file_from_google_drive(file_id, destination)
    os.system('mkdir -p ../data')
    os.system('unzip -o ' + destination + " -d ../data")
    os.system('rm ' + destination)
