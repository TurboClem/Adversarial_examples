mkdir -p ~/workspace/data
cd ~/workspace/data

wget -O EuroSAT_MS.zip "https://zenodo.org/records/7711810/files/EuroSAT_MS.zip"
unzip -q EuroSAT_MS.zip
rm EuroSAT_MS.zip

wget -O EuroSAT_RGB.zip "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"
unzip -q EuroSAT_RGB.zip
rm EuroSAT_RGB.zip
