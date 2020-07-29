# Appearance Fabrication
## Extraction

This is an application that uses multiple input images to extract reflectance values and the corresponding geometric information for different points of an illuminated cylindical surface. 

### Installation
Please install all Python requirements from the * requirements.txt * file. Additionally copy all files from the executables folder into your * C:/Windows * folder for Windows machines. For Linux machines install all frameworks from the executables folder via command line.

### Input 

All input data for the estimation is located in **captures**. It contains a folder for every single estimation. Every of these folders contains a parameter file named **params.yaml**. Furthermore, all needed images for the computation are inside the folder. The parameter file should contain the following parameters: 
* _dict_idx_ is the Aruco dictionary that should be used.
* _lamp_dist_ is the distance between the cylinder center line and the lamp position in millimeter.
* _cyl_rad_ is the radius of the used cylinder in millimeter.
* _marker_size_ is the side length of a single marker in millimeter.
* _slice_start_ is the start 
* _slice_end_ is the 
* _lamp_img_ is the name of the image for estimating the lamp position. 
* _ref_img_ is the name of the image for extracting the reference values. 
* _samp_img_ is the name of the image for extracting the reflectance values of the sample.

### Output

The application will generate several images during the computation, which can be used for debugging purposes. The following images are created for every of the input images: 
* An image which shows the detected and used markers on the cylinder. 
* An image which shows the computed cylinder center. 
* An image which shows the sampled points on the cylinder surface. 
* Multiple images which show plots of the normalized illuminance. 
* Multiple images which show plots of the unnormalized illuminance. 

Additionally, a binary file named **values.bin** is created, which contains all extracted values. The first value is an integer which tells us the total count of sampled points. After, the extracted data one sampled point are written at once. This is repeated until the data for all points were written. The file looks as following: 

* Count of sampled points (_integer_)
* For every sampled point
	* Illuminance (_double_)
	* Theta in (_double_)
	* Theta out (_double_)
	* Theta h (_double_)
	* Theta d (_double_)
	* Phi in (_double_)
	* Phi out (_double_)
	* Phi h (_double_)
	* Phi d (_double_)
	* Delta phi (_double_)

### Execution

By calling _estimate_BRDF.py_ the output for every single folder in captures is computed.