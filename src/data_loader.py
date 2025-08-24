import kagglehub
import pandas as pd
import os
import shutil

#  KaggleHub-Download (wie bisher)
path = kagglehub.dataset_download("somesh24/sea-level-change")
#print("Path to dataset files:", path)

#  Cache-Verzeichnis prüfen
dataset_path = os.path.expanduser("~/.cache/kagglehub/datasets/somesh24/sea-level-change/versions/1")
##print("Dateien im Cache:", os.listdir(dataset_path))


# 3️⃣ CSV-Datei laden
def load_sea_level_csv():
    file_name = "sea_levels_2015.csv"  # anpassen, falls anders
    file_path = os.path.join(dataset_path, file_name)
    df = pd.read_csv(file_path)

    # ⃣ Kopieren in Projektbaum (data/raw)
    project_data_path = os.path.join(os.path.dirname(__file__), "../data/raw")
    os.makedirs(project_data_path, exist_ok=True)
    dest_path = os.path.join(project_data_path, file_name)
    shutil.copy(file_path, dest_path)
    #print(f"Datei kopiert nach: {dest_path}")

    return df


# Laden & kopieren aufrufen
df = load_sea_level_csv()
#print(df.head())
