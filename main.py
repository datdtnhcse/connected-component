import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import label
from PIL import Image
from matplotlib.patches import Rectangle

# Load the input image using Pillow
img = Image.open('input/input_3.png')

# Convert the image to grayscale and then to a numpy array
img_gray = img.convert('L')
img_arr = np.array(img_gray)

def bfs_index(binary_image):
    # Generate a binary structure for connectivity
    struct = np.ones((3, 3), dtype=bool)

    
    # Label connected components in the binary image
    labeled, n = label(binary_image, struct)
    
    # Initialize the component index to 0
    index = np.zeros_like(binary_image) 
    
    # Initialize a list to store the bounding boxes
    boxes = []
    
    # Iterate over each connected component and assign it an index
    for i in range(1, n+1):
        # Extract the binary image of the current component
        component = np.zeros_like(binary_image)
        component[labeled == i] = 1
        
        # Perform BFS on the binary image to index the component
        queue = []
        visited = np.zeros_like(binary_image)
        start = np.unravel_index(np.argmax(component), component.shape)
        queue.append(start)
        visited[start] = 1
        index[start] = i
        while queue:
            current = queue.pop(0)
            for neighbor in np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]) + current:
                if (np.logical_and(neighbor[0] >= 0, neighbor[0] < binary_image.shape[0]) 
                    and np.logical_and(neighbor[1] >= 0, neighbor[1] < binary_image.shape[1]) 
                    and np.all(component[neighbor]) 
                    and not np.all(visited[neighbor])):
                    queue.append(neighbor)
                    visited[neighbor] = 1
                    index[neighbor] = i
        
        # Compute the bounding box of the current component
        rows, cols = np.where(component == 1)
        x1, y1 = np.min(cols), np.min(rows)
        x2, y2 = np.max(cols), np.max(rows)
        boxes.append(((x1, y1), (x2, y2)))
    
    # Create a copy of the input binary image and replace each pixel with its corresponding index
    indexed_image = np.zeros_like(binary_image)
    for i in range(1, n+1):
        indexed_image[labeled == i] = index[labeled == i]
    
    return indexed_image, boxes

indexed_image, boxes = bfs_index(img_arr)

# Display the input image with bounding boxes and indices
fig, ax = plt.subplots()
ax.imshow(img_arr, cmap='gray')
for i, box in enumerate(boxes):
    rect = Rectangle(box[0], box[1][0] - box[0][0], box[1][1] - box[0][1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(box[0][0], box[0][1], str(i+1), color='r', fontsize=8)
plt.show()
