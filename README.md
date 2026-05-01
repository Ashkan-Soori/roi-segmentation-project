## ROI Segmentation Project

This project presents a classical image processing pipeline for Region of Interest (ROI) segmentation in histopathology images.

The main objective is to automatically extract tissue regions by applying grayscale thresholding and basic morphological operations. Instead of relying on deep learning models, the approach focuses on simplicity, interpretability, and reproducibility.

Because the segmentation is based on image intensity rather than color-specific features, the method works consistently for both H&E and IHC stained images.

The purpose of this project is not to achieve state-of-the-art performance, but to demonstrate how fundamental image processing techniques can effectively solve a basic medical image segmentation task in a clear and structured way.

This repository was developed as part of the final examination for the Software and Computing course in the Applied Physics curriculum at the University of Bologna.

- - -

## What the Project Does

The segmentation pipeline follows a clear and structured sequence of steps:

- Loading the input image  
- Converting it to grayscale  
- Applying thresholding (either Otsu or manual)  
- Cleaning the mask using morphological operations  
- Displaying the intermediate and final results  
- Automatically saving the output image  

The overall idea behind this project is to show how classical image processing techniques can effectively solve a basic segmentation task without relying on deep learning models.

- - -

## Installation & Setup

This project has been tested with Python 3.10.11 on Windows, macOS, and Linux.
Before starting, make sure Python and Git are installed on your system.

You can check your Python installation by running:
python --version
python3 --version
git --version
If Python is not installed, download it from:
https://www.python.org/downloads/

- - -

1. Clone the Repository

Open your terminal and navigate to the directory where you want to download the project.

Then run:
```bash

git clone https://github.com/Ashkan-Soori/roi-segmentation-project.git

```
Move into the project folder:
```bash

cd roi-segmentation-project

```


2. Create a Virtual Environment

Although not mandatory, it is strongly recommended to create a virtual environment to keep dependencies isolated from your system Python installation.

On macOS / Linux:

```bash

python3 -m venv roi_env
source roi_env/bin/activate

```

On Windows:

```bash

python -m venv roi_env
roi_env\Scripts\activate

```

After activation, your terminal should show:

(roi_env)
This indicates that the virtual environment is active


3. Install Required Dependencies

```bash

pip install -r requirements.txt

```
This will install all necessary libraries including:


numpy
opencv-python
matplotlib
pytest
coverage

```Markdown

The `requirements.txt` file contains all installable dependencies required to run the application, execute tests, and generate coverage reports.

```

4. Verify Installation

```bash

python -m pytest -v

```

5. Run Tests with Coverage 

```bash

python -m coverage run --source=roi_segmentation -m pytest
python -m coverage report

```
6. Run the Application

Default mode (Otsu thresholding):

```bash

python -m roi_segmentation.main --image data/0_1009_0_0_0.jpg

```
Manual thresholding example:

```bash

python -m roi_segmentation.main --image data/0_1009_0_0_0.jpg --method manual --threshold 180

```

## Running the Tests
Before running the tests, make sure:

You are inside the root directory of the project
All dependencies are installed (see Installation section)

To execute the full test suite, run the following command in your terminal:

```bash

python -m pytest -v

```

This will automatically discover and execute all test files inside the tests/ directory.

If everything is working correctly, you should see an output similar to:
```Markdown

9 passed in 0.xx seconds

```
This confirms that all components of the segmentation pipeline are functioning as expected


If everything is correctly installed, you should see all tests passing successfully.


1. Running Tests with Coverage

To evaluate how much of the source code is covered by the tests, run:

```bash

python -m coverage run --source=roi_segmentation -m pytest
python -m coverage report

```
This will display a coverage summary in the terminal, including:

Total number of statements
Number of missed lines
Overall coverage percentage

For a detailed HTML report, run:

```bash

python -m coverage html

```

```bash

start htmlcov/index.html

```
On Windows, use start htmlcov/index.html to open the report in your browser.

```Markdown

On Windows:
    start htmlcov/index.html

On macOS:
    open htmlcov/index.html

On Linux:
    xdg-open htmlcov/index.html

```


in your browser to inspect line-by-line coverage details.

Coverage Status

The test suite currently achieves 96% code coverage for the roi_segmentation module.

All core segmentation logic (image loading, thresholding, morphology, and main execution flow) is covered by tests.
The small percentage of uncovered lines corresponds to minor branches or visualization-related code.

- - -

 Test Environment

```Markdown

This project was tested on:

- Python 3.10.11
- Windows 10
- macOS

```

## Limitations and Notes

 please consider the following limitations:

1. Supported Image Formats

Only common image formats such as .jpg, .jpeg, and .png are supported.

Other formats may require manual conversion before processing.

All output masks are saved in .png format.

2. Intensity-Based Segmentation

The segmentation pipeline relies purely on grayscale intensity thresholding.

It does not use color-aware processing or machine learning models.

Performance may vary depending on image contrast, lighting conditions, or staining variability.

3. No Learning or Model Adaptation

This project does not include training, fine-tuning, or adaptive mechanisms.

It is a fully classical image processing implementation intended for educational purposes.

4. Binary Segmentation Only

The pipeline produces a binary mask distinguishing tissue from background.

Multi-class segmentation is not supported.

5. Command-Line Interface (CLI) Only

The application is designed to run from the command line.

No graphical user interface (GUI) is currently provided.

6. Sensitivity to Manual Threshold Selection

When using manual thresholding, the segmentation quality depends heavily on the selected threshold value.

An inappropriate threshold may lead to over-segmentation or under-segmentation.

- - -

## Image Attribution

The sample images used in this project were obtained from the publicly available IHC4BC – Compressed Dataset (HER2 subset) on Kaggle:

https://www.kaggle.com/datasets/akbarnejad1991/ihc4bc-compressed

The dataset is publicly accessible and is used in this project strictly for academic and educational purposes.

All rights and credits belong to the original dataset contributors.
For full licensing details and usage terms, please refer to the official Kaggle dataset page.

- - -

## Example Output

Below is an example of the full segmentation pipeline applied to a sample HER2 histopathology image.

The figure displays all main processing stages side by side:

The original RGB image
The grayscale conversion
The raw binary mask obtained after thresholding
The cleaned mask after morphological operations

These outputs allow the user to visually compare the effect of thresholding and morphology on the final ROI segmentation result.

The examples below show results using both the default Otsu method and manual thresholding.


### Otsu Method

<img src="outputs/Otsu_threshold.png" width="800"/>

### Manual Threshold (180)

<img src="outputs/Manual_threshold_180.png" width="800"/>

- - -

## License

This project is licensed under the [MIT License](LICENSE).





