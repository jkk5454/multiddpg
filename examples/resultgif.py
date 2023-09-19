import imageio
import glob

# Path to the folder containing the images
folder_path = '/home/clothsim/multiddpg/data/train_square/test'  # Change this to the path to your folder

# Pattern to match the files
pattern = f'{folder_path}/Test_*_top.png'

# Get all files matching the pattern in the specified folder
filenames = sorted(glob.glob(pattern))

# Create a list to store the images
images = []

# Loop through the files and append them to the images list
for filename in filenames:
    images.append(imageio.imread(filename))
    

# Save the images as a GIF
imageio.mimsave('output.gif', images, duration=0.5)
