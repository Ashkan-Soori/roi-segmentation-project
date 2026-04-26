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



ROI Segmentation Project

This project implements a simple Region of Interest (ROI) segmentation pipeline using classical image processing techniques.

The main goal is to extract tissue regions from histopathology images by applying thresholding and basic morphological operations. The method is fully intensity-based, which means it works for both H&E and IHC images without relying on color-specific assumptions or deep learning models.

This repository was developed for the final examination of the Software and Computing course in the Applied Physics curriculum at the University of Bologna.

Project Idea

Instead of using deep learning, this project demonstrates how far we can go using classical image processing.

The pipeline is intentionally simple and transparent. Every step is visible and understandable:

Load the input image
Convert it to grayscale
Apply thresholding (Otsu or manual)
Clean the mask using morphology
Display intermediate results
Save the final segmentation automatically

The focus is clarity, reproducibility, and structured code organization.

Installation

Make sure Python 3.9 or newer is installed.

All required libraries are listed in requirements.txt.

From the root directory of the project, run:

pip install -r requirements.txt
Project Structure
roi-segmentation-project/
│
├── data/                  # Input images
├── roi_segmentation/      # Main segmentation module
├── tests/                 # Unit tests
├── Outputs/               # Saved segmentation results
├── requirements.txt
└── README.md
How to Run the Project

Open a terminal (CMD on Windows) and make sure you are inside the root folder of the project.

Default Mode (Otsu Thresholding)
On Windows
python -m roi_segmentation.main --image data\0_1009_0_0_0.jpg
On macOS/Linux
python3 -m roi_segmentation.main --image data/0_1009_0_0_0.jpg

Otsu thresholding automatically determines the best threshold value based on the image histogram.

Manual Thresholding

If you want to manually control the threshold:

On Windows
python -m roi_segmentation.main --image data\0_1009_0_0_0.jpg --method manual --threshold 180
On macOS/Linux
python3 -m roi_segmentation.main --image data/0_1009_0_0_0.jpg --method manual --threshold 180

Manual mode allows experimenting with different threshold values depending on image contrast.

Available Arguments
Argument	Description
--image	Path to the input image (required)
--method	"otsu" (default) or "manual"
--threshold	Threshold value (used only in manual mode)
--kernel	Kernel size for morphological operations
Output

When the script runs, it displays four images:

Original image
Grayscale image
Raw threshold mask
Cleaned mask after morphology

The final result is automatically saved inside the Outputs/ folder.
If the folder does not exist, it will be created automatically.

All output files are saved in PNG format.

**Example Output**



Below is a sample result showing the full processing pipeline (original image, grayscale, raw mask, and cleaned mask):

\### Otsu Method

<img src="outputs/Otsu\_threshold.png" width="800"/>



\### Manual Threshold (180)

<img src="outputs/Manual\_threshold\_180.png" width="800"/>

### Otsu Method
![Otsu Result](outputs/Otsu_threshold.png)

### Manual Threshold (180)
![Manual Result](outputs/Manual_threshold_180.png)