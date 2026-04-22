\# ROI Segmentation Project



This project implements a simple ROI segmentation pipeline using classical image processing methods.



The goal is to extract the main tissue region from an image using thresholding and improve the result using morphological operations.



The pipeline includes:



\- Image loading

\- Grayscale conversion

\- Otsu or manual thresholding

\- Morphological cleaning

\- Result visualization and saving



Installation:



Install the required libraries using:



pip install -r requirements.txt



How to run:



From the root folder of the project, run:



python -m roi\_segmentation.main --image data/your\_image.jpg



Example:



python -m roi\_segmentation.main --image data/0\_1009\_0\_0\_0.jpg



Available options:



\--method otsu (default)

\--method manual

\--threshold 200 (only used with manual mode)



This project focuses on classical image processing techniques rather than deep learning.

