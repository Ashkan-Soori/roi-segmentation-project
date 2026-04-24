ROI Segmentation Project



This project implements a simple Region of Interest (ROI) segmentation pipeline using classical image processing techniques.



The main goal is to extract the tissue region from histopathology images by applying thresholding and basic morphological operations.

It works for both H\&E and IHC images since the segmentation is based on image intensity rather than color-specific processing.



This repository was developed for the final examination of the Software and Computing course in the Applied Physics curriculum at the University of Bologna.



**What the project does**



The segmentation process includes:



Loading the input image

Converting it to grayscale

Applying thresholding (either Otsu or manual)

Cleaning the mask using morphological operations

Displaying the intermediate and final results

Automatically saving the output image



The idea behind this project is to demonstrate how classical image processing methods can be used to solve a basic segmentation task without relying on deep learning.



**Installation**



Make sure Python (3.9 or newer) is installed.



Install the required libraries using:



pip install -r requirements.txt



All required packages are listed inside the requirements.txt file.



**How to Run**



Open a terminal (CMD on Windows) and make sure you are inside the root folder of the project.



Default method (Otsu thresholding)



**On Windows**:



python -m roi\_segmentation.main --image data\\0\_1009\_0\_0\_0.jpg



On macOS/Linux:



python3 -m roi\_segmentation.main --image data/0\_1009\_0\_0\_0.jpg



**Manual thresholding**



On Windows:



python -m roi\_segmentation.main --image data\\0\_1009\_0\_0\_0.jpg --method manual --threshold 180



On macOS/Linux:



python3 -m roi\_segmentation.main --image data/0\_1009\_0\_0\_0.jpg --method manual --threshold 180



**Available Arguments**



\--image : path to the input image (required)

\--method : choose between "otsu" (default) or "manual"

\--threshold : threshold value used only in manual mode

\--kernel : kernel size used for morphological operations



**Output**



When the script runs, it shows four images:



Original image

Grayscale version

Raw threshold mask

Cleaned mask after morphology



The final result is automatically saved inside the Outputs/ folder.

If the folder does not exist, it will be created automatically.



All output images are saved in PNG format.



**Project Structure**



roi-segmentation-project/

* &#x20;data/ # Input images
* &#x20;roi\_segmentation/ # Main segmentation code
* &#x20;tests/ # Basic unit tests
* &#x20;Outputs/ # Saved results
* &#x20;requirements.txt
* &#x20;README.md/
* 

**Notes**



This project performs binary segmentation based on intensity thresholding.

It does not include deep learning methods.



The goal is to provide a simple, clear, and reproducible example of classical image processing applied to medical images.



**Example Output**



Below is a sample result showing the full processing pipeline (original image, grayscale, raw mask, and cleaned mask):



\### Otsu Method

!\[Otsu Result](./image/Otsu\_threshold.png)



\### Manual Threshold (180)

!\[Manual Result](./image/Manual\_threshold\_180.png)

