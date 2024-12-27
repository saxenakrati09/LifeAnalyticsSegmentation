# Life Analytics

This repository reads the images of dishes with cells and their colonies and generates their segmentations.

-------------

### Setting up the Environment
- We use python 3.10 as described in the Github repo of cellpose
`conda create -n cellpose python=3.10 anaconda`
- Once the environment is setup, activate it
`conda activate cellpose `
- Install the libraries
`pip install -r requirements.txt`
-------------
### Running the Code
- Keep the images in **"images"** folder and remove any other files (which are not the image data you want to run the code on)

For running the cellpose model and saving the output .npy file in the **segment** folder
 `cd segment`
`python cyto_model.py`

For saving the masks, and the merged images and masks in the **plot** folder.
` cd plot`
`python plot.pt`


Contributor: Krati Saxena