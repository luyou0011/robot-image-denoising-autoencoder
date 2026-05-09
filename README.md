# Rauschunterdrückung mittels Convolutional Autoencoder

Dieses Projekt befasst sich mit der Eliminierung von Bildrauschen in Roboter-Szenen (RobotMNIST) unter Verwendung von Deep Learning. Ziel ist es, verrauschte Eingabebilder durch ein Encoder-Decoder-Netzwerk zu rekonstruieren.

## Methodik & Architektur
- **Modell:** Denoising Autoencoder basierend auf Convolutional Neural Networks (CNN).
- **Framework:** PyTorch.
- **Architektur:** Der Encoder reduziert die Dimensionalität (Feature Extraction), während der Decoder über Transposed Convolutions das Bild wiederherstellt.
- **Experimente:** Vergleich von drei Modellvarianten (2, 3 und 4 Schichten) sowie Evaluation von Aktivierungsfunktionen (ReLU, Sigmoid, Tanh).

## Ergebnisse
Die besten Ergebnisse wurden durch die Kombination von **ReLU-** und **Sigmoid-Funktionen** erzielt. Mit zunehmender Tiefe des Netzwerks (4 Schichten) konnte eine signifikant höhere Genauigkeit bei der Rekonstruktion feiner Details erreicht werden.

## Projektstruktur
- `main.py`: Zentrales Skript für das Training und die Visualisierung.
- `/model`: Definition der Denoising-Modelle.
- `/utils`: Skripte zur Datenaufbereitung und zum Laden des RobotMNIST-Datensatzes.
- `/Images`: Beispieldaten und grafische Auswertungen (Loss/Accuracy).

## Voraussetzungen
- Python 3.x
- PyTorch / Torchvision
- Matplotlib, NumPy
