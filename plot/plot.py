from skimage.measure import label, regionprops 

import os
import numpy as np
import cv2
import torch
from icecream import ic
import sys
sys.path.append("..")
from utils.utils import extract_filename

if __name__=="__main__":
    output_ = ["_30_cyto3_seg.npy", "_15_cyto3_seg.npy", "_20_cyto3_seg.npy"]
    files = os.listdir("../images/")
    # output_filenames = []
    # for model_type in modelslist:
    for file in files:
        # Add your processing code here
        path = os.path.join('../images/', file) 
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

        result = extract_filename(path)
        output_filenames = [result + x for x in output_] 
        ic(output_filenames)
        # Check if CUDA is available and use GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Convert to Torch tensors
        image_tensor = torch.from_numpy(image).to(device)
        
        # Convert grayscale image to 3-channel image
        image_3channel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Convert to Torch tensor
        image_3channel_tensor = torch.from_numpy(image_3channel).to(device)

        # check if the file exists
        if os.path.exists(os.path.join('../segment/', output_filenames[0])):
            mask1 = np.load(os.path.join('../segment/', output_filenames[0]), allow_pickle=True).item()
            mask1 = mask1['masks']
            # Normalize the mask values to 0-255
            mask_normalized = cv2.normalize(mask1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # Create a red mask
            red_mask = np.zeros((mask1.shape[0], mask1.shape[1], 3), dtype=np.uint8)
            red_mask[:, :, 2] = mask_normalized  # Set red channel
            cv2.imwrite(result + '_' + output_filenames[0] + '.jpg', red_mask)
            print(f"Saved red mask!")
            
            red_mask_tensor = torch.from_numpy(red_mask).to(device)
            # overlayed_image_tensor_red = torch.max(image_3channel_tensor, red_mask_tensor)
        if os.path.exists(os.path.join('../segment/', output_filenames[1])):
            mask2 = np.load(os.path.join('../segment/', output_filenames[1]), allow_pickle=True).item()
            mask2 = mask2['masks']

            # Convert the mask to float32 before normalization 
            mask2_float32 = mask2.astype(np.float32)
            # Normalize the second mask values to 0-255 
            mask2_normalized = cv2.normalize(mask2_float32, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Create a blue mask 
            blue_mask = np.zeros((mask2.shape[0], mask2.shape[1], 3), dtype=np.uint8) 
            blue_mask[:, :, 0] = mask2_normalized # Set blue channel
            
            cv2.imwrite(result + '_' + output_filenames[1] + '.jpg', blue_mask)
            print(f"Saved blue mask!")
            
            blue_mask_tensor = torch.from_numpy(blue_mask).to(device)
            # overlayed_image_tensor_blue = torch.max(image_3channel_tensor, blue_mask_tensor)
        if os.path.exists(os.path.join('../segment/', output_filenames[2])):
            mask3 = np.load(os.path.join('../segment/', output_filenames[2]), allow_pickle=True).item()
            mask3 = mask3['masks']

            # Convert the mask to float32 before normalization 
            mask3_float32 = mask3.astype(np.float32)
            # Normalize the second mask values to 0-255 
            mask3_normalized = cv2.normalize(mask3_float32, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Create a blue mask 
            green_mask = np.zeros((mask3.shape[0], mask3.shape[1], 3), dtype=np.uint8) 
            green_mask[:, :, 1] = mask3_normalized # Set blue channel
            
            cv2.imwrite(result + '_' + output_filenames[2] + '.jpg', green_mask)
            print(f"Saved green mask!")
            
            green_mask_tensor = torch.from_numpy(green_mask).to(device)
            # overlayed_image_tensor_green = torch.max(image_3channel_tensor, green_mask_tensor)

        
        


        # Overlay the mask
        # overlayed_image_tensor = torch.max(image_3channel_tensor, red_mask_tensor)
        # final_overlayed_image_tensor = torch.max(overlayed_image_tensor, blue_mask_tensor)
        # final_overlayed_image_tensor = torch.max(final_overlayed_image_tensor, green_mask_tensor)
        # Initialize with the original image tensor
        final_overlayed_image_tensor = image_3channel_tensor

        # Check for the existence of each mask tensor and apply accordingly
        if 'red_mask_tensor' in locals():
            final_overlayed_image_tensor = torch.max(final_overlayed_image_tensor, red_mask_tensor)

        if 'blue_mask_tensor' in locals():
            final_overlayed_image_tensor = torch.max(final_overlayed_image_tensor, blue_mask_tensor)

        if 'green_mask_tensor' in locals():
            final_overlayed_image_tensor = torch.max(final_overlayed_image_tensor, green_mask_tensor)

        # Label each cluster 
        # labeled_clusters = label(mask2)

        final_overlayed_image_tensor = final_overlayed_image_tensor.cpu().numpy()
        final_overlayed_image_tensor = final_overlayed_image_tensor.astype(np.uint8)

        # Add labels to the clusters 
        # for region in regionprops(labeled_clusters): 
        #     # Take the centroid of the region to place the label 
        #     centroid = region.centroid 
        #     cv2.putText(final_overlayed_image_tensor, str(region.label), (int(centroid[1]), int(centroid[0])), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 4, cv2.LINE_AA)
    
        # # Save the output image
        cv2.imwrite(result + '_overlayed_image.jpg', final_overlayed_image_tensor)

        print("Overlay completed and saved as overlayed_image.jpg")
