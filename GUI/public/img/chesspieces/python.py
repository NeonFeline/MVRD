import os
from PIL import Image

def replace_colors(image_path, output_path, color1, color2):
    img = Image.open(image_path).convert("RGBA")
    pixels = img.load()
    
    for y in range(img.height):
        for x in range(img.width):
            r, g, b, a = pixels[x, y]
            
            if (r, g, b) == (255, 255, 255):  # White
                pixels[x, y] = (*color1, a)
            elif (r, g, b) == (0, 0, 0):  # Black
                pixels[x, y] = (*color2, a)
    
    img.save(output_path)

def process_images_in_folder(folder_path, output_folder, color1, color2):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        
        # Only process files that are images (optional, based on file extension)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Set the output path for the modified image
            output_path = os.path.join(output_folder, f"{filename}")
            # Apply the color replacement function
            replace_colors(image_path, output_path, color1, color2)

# Example usage
folder_path = "./save/wikipedia"
output_folder = "./wikipedia"
color2 = (95, 61, 196)  # Red
color1 = (243, 240, 255)  # Green

process_images_in_folder(folder_path, output_folder, color1, color2)

