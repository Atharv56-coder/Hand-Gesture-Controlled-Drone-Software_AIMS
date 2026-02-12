import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os

input_base = 'my_custom_gestures'
output_base = 'augmented_dataset'
os.makedirs(output_base, exist_ok=True)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,   
    fill_mode='nearest'     
)

for folder in os.listdir(input_base):
    folder_path = os.path.join(input_base, folder)
    if not os.path.isdir(folder_path): continue

    save_path = os.path.join(output_base, folder)
    os.makedirs(save_path, exist_ok=True)

    images = os.listdir(folder_path)
    copies_per_image = 500 // len(images)

    print(f"Augmenting {folder}...")

    for img_name in images:
        img = load_img(os.path.join(folder_path, img_name))
        x = img_to_array(img)        
        x = x.reshape((1,) + x.shape) 

        i = 0
        for batch in datagen.flow(x, batch_size=1, 
                                  save_to_dir=save_path, 
                                  save_prefix='aug', 
                                  save_format='jpg'):
            i += 1
            if i >= copies_per_image:
                break 