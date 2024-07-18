import os
from PIL import Image

def convert_and_resize_image(input_folder, output_folder, size=(64, 64)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert('L') 
            img = img.resize(size)  

            output_path = os.path.join(output_folder, filename)
            img.save(output_path)

            print(f"Processed {filename}")
    print("done")


input_folder = "mazes/mazes_data"  
output_folder = "mazes_64/mazes_64_data" 

convert_and_resize_image(input_folder, output_folder, size=(64, 64))
