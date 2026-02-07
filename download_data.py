import os
import zipfile
import gdown
from pathlib import Path
from src.config import CONFIG

ARCHIVE_NAME = "data.zip"
GDRIVE_FILE_ID = "1Orlk5QBETGj8TP-MXlvIM43wO9QQ6tym"

def setup_workspace():
    root_dir = CONFIG["root_dir"]
    data_dir = CONFIG["data_dir"]
    
    data_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Setup] Проверка данных в: {data_dir}")

    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"[Setup] Данные уже распакованы. Пропуск.")
        return

    archive_path = root_dir / ARCHIVE_NAME
    if not archive_path.exists():
        print(f"[Setup] Скачивание {ARCHIVE_NAME}...")
        url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
        gdown.download(url, str(archive_path), quiet=False)
    
    print(f"[Setup] Распаковка архива...")
    try:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(root_dir)
        print("[Setup] Распаковка завершена успешно.")
        
        os.remove(archive_path)
    except Exception as e:
        print(f"[Error] Ошибка распаковки: {e}")

if __name__ == "__main__":
    setup_workspace()