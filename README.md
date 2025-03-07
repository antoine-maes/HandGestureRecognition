# HandGestureRecognition

Projet de Hand Gesture Recognition avec PyTorch et des images de Leap Motion


## Dataset

Utilisation du dataset Leap Motion de Briareo : <https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=31>

Ce dataset contient des images de mains en leap motion ainsi que des json pour les landmarks.

000 --> 025 : chaque dossier = une personne
g00 --> g12_test : chaque dossier = un geste
00 --> 02 : chaque dossier = répetition du geste

## Scripts

- ''/visu.py'' : Ce script charge le dataset de gestes de la main, puis visualise et anime ces séquences en gif.

- ''/model/main.ipynb'' : Création et entraîment d'un GCN sur les images du Leap Motion. 

## Modèle d'IA

Modèle d'IA GCN entraîné : 'model/best_leap_model.pth'

Performances du modèle : 
- Train Loss: 0.0768, Train Acc: 97.30%
- Val Loss: 0.3004, Val Acc: 89.14%
- Meilleure précision de validation: 91.40%

D'autres indicateurs d'évaluations sont dans le notebook.



5e année ISEN Lille avec Hazem Wannous

Louis Lecouturier, Antoine Maes, Nicolas Broage