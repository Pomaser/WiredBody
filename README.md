# WiredBody — Face & Hand Detector

Real-time detekce obličeje, očí, úst a drátěné ruky z webkamery.

## Spuštění (vývojové prostředí)

```powershell
& "C:\Program Files\Anaconda3\python.exe" face_features_detector.py
```

S konkrétní kamerou:
```powershell
& "C:\Program Files\Anaconda3\python.exe" face_features_detector.py --camera 1
```

Výpis dostupných kamer:
```powershell
& "C:\Program Files\Anaconda3\python.exe" face_features_detector.py --list
```

## Ovládání

| Klávesa | Akce |
|---------|------|
| `Q` | Ukončit |
| `O` | Zobrazit / skrýt texty (Faces, Hands, atd.) |
| `F` | Přepnout fullscreen / okno |

## Konfigurace kamery

Uprav `config.json`:
```json
{
    "camera": {
        "preferred": 1,
        "fallback": [0, 2]
    }
}
```

Pokud předvolená kamera není dostupná, použije se první dostupná z `fallback`.

## Sestavení .exe (PyInstaller)

### Předpoklady

```powershell
& "C:\Program Files\Anaconda3\Scripts\pip.exe" install pyinstaller
```

Ověř cestu k Haar cascade souborům:
```powershell
& "C:\Program Files\Anaconda3\python.exe" -c "import cv2; print(cv2.data.haarcascades)"
```

### Build příkaz

```powershell
cd "c:\Users\maj9bj\OneDrive - Bosch Group\Projekty\Python\WiredBody"

& "C:\Program Files\Anaconda3\python.exe" -m PyInstaller `
    --onefile `
    --name "FaceDetector" `
    --add-data "glasses_single.png;." `
    --add-data "joint_single.png;." `
    --add-data "fuck_off.wav;." `
    --add-data "config.json;." `
    --add-data "C:\Users\maj9bj\AppData\Roaming\Python\Python39\site-packages\cv2\data\haarcascade_frontalface_default.xml;." `
    --add-data "C:\Users\maj9bj\AppData\Roaming\Python\Python39\site-packages\cv2\data\haarcascade_smile.xml;." `
    --collect-all mediapipe `
    face_features_detector.py
```

Výsledek: `dist\FaceDetector.exe`

### Distribuce

Vedle `FaceDetector.exe` lze umístit vlastní `config.json` pro přenastavení kamery bez nutnosti nového buildu.

### Diagnostika

Pokud exe spadne, zkontroluj `facedetector.log` vedle exe souboru.
Pro zobrazení chyb v konzoli odstraň z build příkazu příznak `--noconsole`.
