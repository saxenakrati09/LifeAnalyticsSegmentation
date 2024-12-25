import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io
from cellpose.io import imread
import skimage
import os
from icecream import ic
import sys
sys.path.append("..")
from utils.utils import extract_filename
io.logger_setup()

def generate_model_results(img, diam, channels, outputnpy, model_type='cyto3', use_gpu=True, flow_threshold = 0.4, do_3D=False):
    
    # model_type='cyto' or 'nuclei' or 'cyto2' or 'cyto3'
    """
    Run the Cellpose model on an image and save the results to a npy file.

    Parameters
    ----------
    img : image
        The image for segmentation.
    diam : int
        The diameter of the cells in the image.
    channels : list of int
        The channels to segment (e.g., [0, 0] for grayscale or [2, 3] for RGB).
    outputnpy : str
        The path to save the npy file containing the masks and flows.
    model_type : str, optional
        The type of model to use (default is 'cyto3').
    use_gpu : bool, optional
        Whether to use the GPU (default is True).
    flow_threshold : float, optional
        The threshold for the flow (default is 0.4).
    do_3D : bool, optional
        Whether to do 3D segmentation (default is False).
    """

    model = models.Cellpose(model_type=model_type, gpu=use_gpu)
    masks1, flows1, styles1, diams1 = model.eval(img, diameter=diam, channels=channels, flow_threshold=flow_threshold, do_3D=False)
    # Flip the image horizontally 
    image_flipped = np.fliplr(img)
    # Run the model on the flipped image 
    masks2, flows2, styles2, diams2 = model.eval(image_flipped, diameter=diam, channels=channels, flow_threshold=flow_threshold, do_3D=False)
    # Flip the masks back to the original orientation 
    masks2 = np.fliplr(masks2) 
    # Combine the masks (using logical OR to merge) 
    merged_masks = np.logical_or(masks1, masks2).astype(np.uint8)

    # Merge the flow fields (averaging the flow fields) 
    flows2_flipped = [np.fliplr(flow) for flow in flows2]
    merged_flows = [(flow1 + flow2_flipped) / 2 for flow1, flow2_flipped in zip(flows1, flows2_flipped)]

    io.masks_flows_to_seg(img, merged_masks, merged_flows, outputnpy, channels=channels, diams=diam)
    # io.save_masks(img, merged_masks, merged_flows, pngoutput, png=True)

if __name__=="__main__":
    files = os.listdir("../images_dummy/")
    diameters = [30, 15, 10] # 
    # modelslist = ["cyto3"]

    model_type = 'cyto3'
    for file in files:
        for diam in diameters:
        # for model_type in modelslist:
        
            ic(f"Processing file {file} with model {model_type} and diameter {diam}")
            # Add your processing code here
            path = os.path.join('../images_dummy/', file) 
            img = io.imread(path)

            result = extract_filename(path)

            channels = [[0,0]]
            outputname = result + "_" + str(diam) + "_" + str(model_type)

            generate_model_results(img, diam, channels, outputname, model_type=model_type, use_gpu=True, flow_threshold = 0.4, do_3D=False)