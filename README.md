# SMS Spam Classifier - CNN

Projekt klasyfikuje wiadomości SMS jako **spam** lub **ham** z użyciem sieci CNN i optymalizacji modelu (pruning + kwantyzacja).

## 1. Wymagania
- Windows 10/11 (64-bit)
- Python 3.11
- NVIDIA GPU z CUDA  lub CPU

## 2. Instalacja Python i TensorFlow

```cmd
python -m venv C:\tfvenv
C:\tfvenv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

