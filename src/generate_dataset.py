import numpy as np
import cv2
import os

# Function to create a dataset of custom geometric shapes
def create_shape_dataset(samples_per_shape=500, dataset_dir='../generated_dataset'):
    shape_types = ['line', 'circle', 'ellipse', 'rectangle', 'rounded_rect', 'polygon', 'star']
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        
    
    for shape in shape_types:
        shape_folder = os.path.join(dataset_dir, shape)
        if not os.path.exists(shape_folder):
            os.makedirs(shape_folder)
        
        for index in range(samples_per_shape):
            canvas = np.zeros((128, 128), dtype=np.uint8)
            
            if shape == 'line':
                x_start, y_start = np.random.randint(0, 128, size=2)
                x_end, y_end = np.random.randint(0, 128, size=2)
                cv2.line(canvas, (x_start, y_start), (x_end, y_end), 255, 2)
            
            elif shape == 'circle':
                rad = np.random.randint(10, 40)
                x_center, y_center = np.random.randint(rad, 128-rad, size=2)
                cv2.circle(canvas, (x_center, y_center), rad, 255, 2)
            
            elif shape == 'ellipse':
                x_center, y_center = np.random.randint(20, 108, size=2)
                axes_length = np.random.randint(10, 50, size=2)
                rot_angle = np.random.randint(0, 180)
                cv2.ellipse(canvas, (x_center, y_center), tuple(axes_length), rot_angle, 0, 360, 255, 2)
            
            elif shape == 'rectangle':
                x_start, y_start = np.random.randint(0, 108, size=2)
                x_end, y_end = x_start + np.random.randint(20, 40), y_start + np.random.randint(20, 40)
                cv2.rectangle(canvas, (x_start, y_start), (x_end, y_end), 255, 2)
            
            elif shape == 'rounded_rect':
                x_start, y_start = np.random.randint(0, 108, size=2)
                x_end, y_end = x_start + np.random.randint(20, 40), y_start + np.random.randint(20, 40)
                radius = np.random.randint(5, 15)
                cv2.rectangle(canvas, (x_start, y_start), (x_end, y_end), 255, 2)
                cv2.circle(canvas, (x_start, y_start), radius, 255, 2)
                cv2.circle(canvas, (x_end, y_end), radius, 255, 2)
                cv2.circle(canvas, (x_start, y_end), radius, 255, 2)
                cv2.circle(canvas, (x_end, y_start), radius, 255, 2)
            
            elif shape == 'polygon':
                vertices = np.random.randint(0, 128, size=(5, 2))
                cv2.polylines(canvas, [vertices], isClosed=True, color=255, thickness=2)
            
            elif shape == 'star':
                # Simplified star shape
                vertices = np.random.randint(0, 128, size=(5, 2))
                cv2.polylines(canvas, [vertices], isClosed=True, color=255, thickness=2)
            
            file_path = os.path.join(shape_folder, f'{index}.png')
            cv2.imwrite(file_path, canvas)

# Generate the custom dataset
create_shape_dataset()
