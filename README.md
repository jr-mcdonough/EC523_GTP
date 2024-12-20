# EC523_GTP

This is an implementation of the graph transformer model for pathology classification from:
https://github.com/vkola-lab/tmi2022/tree/main

No data is uploaded here, but the file structure is preserved. If a directory of training image patches and test image patches is added, all the code can be run.

.csv files are used throughout to manage loading the data; sample code to write a set of filepaths to a .csv file is included in data_prepare.ipynb.

To train the graph transformer model end-to-end:

# 1. Download data
The data comes from a Kaggle challenge: https://www.kaggle.com/c/prostate-cancer-grade-assessment/data

get_kaggle_data.ipynb can be used to download the data or a subset of it; the notebook can be run using a kaggle notebook

# 2. Patch images
Images are patched using the original implementation's code; tmi2022/src/tile_WSI. Clone this GitHub, ensure OpenSlide is installed, and run the script using the data downloaded from Kaggle

# 3. Train feature extractor
With the image patches uploaded, and their path set in config.yaml, the feature extractor can be trained. Run the block in run_GTP.ipynb, or copy the command from this block and run in terminal.

# 4. Build graphs
When the feature extractor is trained, run the command in run_GTP.ipynb to build graphs. The parameters are set do load and save the patches and graphs, but can be modified to save/load to and from different locations, or to load either training or test patches

# 5. Train graph transformer
When graphs are prepared, the classifier model can be trained with the command in run_GTP.ipynb

# 6. Test the model, evaluate performance
Checkpoints from training, and the best validation model, are saved. data_prepare.ipynb contains some options for evaluating/visualizing the model's performance. Ensure that test data graphs are built.
