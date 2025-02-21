# Importations nécessaires
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


# Définition de la classe BriareoDataset
class BriareoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sequences = []  # Au lieu de samples, on stocke des séquences
        
        print(f"Chargement du dataset depuis {root_dir}")

        for person_id in range(0, 26):
            person_dir = f"{str(person_id).zfill(3)}"
            person_path = os.path.join(root_dir, person_dir)
            
            if not os.path.exists(person_path):
                continue
                
            for gesture_id in range(12):
                gesture_dir = f"g{str(gesture_id).zfill(2)}"
                gesture_path = os.path.join(person_path, gesture_dir)
                
                if not os.path.exists(gesture_path):
                    print(f"Skipping {gesture_path}")
                    continue
                
                for repetition_id in range(3):
                    repetition_dir = f"{str(repetition_id).zfill(2)}"
                    repetition_path = os.path.join(gesture_path, repetition_dir)
                    
                    if not os.path.exists(repetition_path):
                        print(f"Skipping {repetition_path}")
                        continue
                    
                    sequence = {
                        'person_id': person_id,
                        'gesture_id': gesture_id,
                        'repetition_id': repetition_id,
                        'frames': []
                    }
                    
                    valid_sequence = True
                    for frame_id in range(40):
                        l_img_path = os.path.join(repetition_path, 'L', 'raw', f"{str(frame_id).zfill(3)}_rl.png")
                        r_img_path = os.path.join(repetition_path, 'R', 'raw', f"{str(frame_id).zfill(3)}_rr.png")

                        
                        if os.path.exists(l_img_path) and os.path.exists(r_img_path):
                            sequence['frames'].append({
                                'frame_id': frame_id,
                                'l_img_path': l_img_path,
                                'r_img_path': r_img_path
                            })
                        else:
                            valid_sequence = False
                            print(f"Séquence incomplète : {l_img_path} ou {r_img_path} manquant")
                            break
                    
                    if valid_sequence and len(sequence['frames']) ==  40:
                        self.sequences.append(sequence)


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Préparer la liste pour stocker toutes les images de la séquence
        sequence_images = []
        
        # Charger les 40 paires d'images
        for frame in sequence['frames']:
            l_image = Image.open(frame['l_img_path']).convert('L')
            r_image = Image.open(frame['r_img_path']).convert('L')
            
            if self.transform:
                l_image = self.transform(l_image)
                r_image = self.transform(r_image)
            
            # Ajouter chaque paire d'images comme un tuple
            sequence_images.append((l_image, r_image))
        
        return {
            'images': sequence_images,  # Liste de 40 tuples (image_gauche, image_droite)
            'gesture_id': sequence['gesture_id'],
            'person_id': sequence['person_id'],
            'repetition_id': sequence['repetition_id']
        }
    


# Définir les transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Créer l'instance du dataset
dataset = BriareoDataset(root_dir='leap_motion/train', transform=transform)

def visualize_sequence(sequence_list, save_gif=False):
    # Créer la figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Normaliser les images pour un meilleur affichage
    l_image, r_image = sequence_list[0]
    l_image = l_image.numpy().squeeze()
    r_image = r_image.numpy().squeeze()
    
    # Normalisation manuelle des images
    l_image = (l_image - l_image.min()) / (l_image.max() - l_image.min())
    r_image = (r_image - r_image.min()) / (r_image.max() - r_image.min())
    
    # Initialisation des images
    l_img = ax1.imshow(l_image, cmap='gray')
    r_img = ax2.imshow(r_image, cmap='gray')
    
    ax1.set_title('Image gauche')
    ax2.set_title('Image droite')
    ax1.axis('off')
    ax2.axis('off')
    
    def update(frame):
        # Récupérer et normaliser les images du frame courant
        l_frame, r_frame = sequence_list[frame]
        l_frame = l_frame.numpy().squeeze()
        r_frame = r_frame.numpy().squeeze()
        
        # Normalisation manuelle
        l_frame = (l_frame - l_frame.min()) / (l_frame.max() - l_frame.min() + 1e-8)
        r_frame = (r_frame - r_frame.min()) / (r_frame.max() - r_frame.min() + 1e-8)
        
        l_img.set_array(l_frame)
        r_img.set_array(r_frame)
        return [l_img, r_img]
    
    # Créer l'animation
    anim = FuncAnimation(fig, update, 
                        frames=len(sequence_list),
                        interval=50,
                        blit=True,
                        repeat=True)
    
    if save_gif:
        writer = PillowWriter(fps=20)
        anim.save('gesture_sequence.gif', writer=writer)
    
    plt.tight_layout()
    plt.show()


# Visualiser la séquence
print("Chargement de la séquence...")
visualize_sequence(dataset[0]['images'], save_gif=True)