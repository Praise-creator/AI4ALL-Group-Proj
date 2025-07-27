import kaggle
import os
import zipfile

os.makedirs('data', exist_ok=True)

print("Downloading dataset...")
kaggle.api.competition_download_files(
    'optiver-realized-volatility-prediction', 
    path='./data'
)

zip_path = './data/optiver-realized-volatility-prediction.zip'
if os.path.exists(zip_path):
    print("Unzipping dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('./data')
    print("Dataset downloaded and extracted to ./data folder!")
else:
    print("Download completed, but zip file not found. Check ./data folder.")
