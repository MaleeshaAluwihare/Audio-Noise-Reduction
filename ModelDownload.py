import requests

url = "https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet2.zip"
response = requests.get(url)

with open("DeepFilterNet2.zip", "wb") as f:
    f.write(response.content)
