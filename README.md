# PoxelNet
A VoxelNet like approach for Shape Classification that works directly on PointClouds

## Requirements
The user needs to have GPU enabled machine with Linux Environment in order to run MinkowskiEngine effectively. 

### System Requirements
- Python
- Linux Machine

### Python Installations
- [Pytorch](https://pytorch.org/get-started/locally/)
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)
- [numpy](https://numpy.org/install/)
- [scikit-learn](https://scikit-learn.org/stable/install.html)  

Optional Requirements for Visualization and creating PointCloud dataset
- [trimesh](https://trimsh.org/)
- [matplotlib](https://matplotlib.org/)

## Original Dataset
The dataset is in the form of 3D CAD models. 

Download the dataset and put it into the ```dataset/``` folder  
Links for downloading - 
- ```ModelNet 40```: http://modelnet.cs.princeton.edu/ModelNet40.zip

The files can be viewed using - 
- ```Notepad```: For seeing vertex and face information
- ```MeshLab```: For seeing the CAD model
- We have also provided helper functions in ```utils.py``` that allows the user to inspect the meshes using ```trimesh```.

## Point Cloud Dataset
PoxelNet works on 3D point clouds and not meshes. Hence, random sampling of the points is required over the 3D meshes. ```create-point_cloud.py``` file does the work of sampling. 
In order to get the PointClouds, the user just needs to run the following command -

`python create_point_cloud.py --path=<Path_to_Original_ModelNet40_dataset> --N=<Number_of_points>`  

The default value of ```N``` is set to 4096

In order to directly get the points, we have provided links to Point Clouds created using Uniform Sampling from the ModelNet40 dataset.
- [Training data with 4096 points](https://drive.google.com/file/d/1IzuMSIg6R56YeggdX5v3LNqgcsTo4N74/view?usp=sharing)
- [Testing data with 4096 points](https://drive.google.com/file/d/1GrO4LRBhfvl5eaOFw1m5b50kB_PpSCM4/view?usp=sharing)
- [Training data with 2048 points](https://drive.google.com/file/d/1hQAaA40xrD_oiOa1uhUDaoiPZCjxTczV/view?usp=sharing)
- [Testing data with 2048 points](https://drive.google.com/file/d/1-25-cqUfOo_f-GZq1Z8VMWCvQa_1U1hG/view?usp=sharing)

## Training
In order to train the model from scratch, the user needs to run the following command - 

```python train.py --path_to_dataset=<path_to_point_cloud_dataset> --store_weights=<path_where_training_weights_will_be_stored>```  

In order to resume training of the model from a previous checkpoint, the user needs to run -    
```python train.py --path_to_dataset=<path_to_dataset> --store_weights=<path_for_storing_weights> --load_pretrained --store_weights=<path_to_previous_checkpoint>```

More optional arguments and their descriptions that can be passed to the functions are included in the file. You can see them using the `--help` flag

## Testing
The testing of the model can be performed via the `test.py` file script. The user needs to have pretrained model stored in order to make the predictions directly without training  
```python test.py --path_to_dataset=<path_to_dataset> --path_weights=<path_to_the_pretrained_model>```


Note: The *path to dataset* should include the path to the directory. The directory then should contain the *train.npy* and *test.npy* files which are provided above. The files include the normalized pointcloud for every Modelnet40 object along with corresponding labels



## Contrinutions
Almost, the entire code of the project is written by - 
- [Fenil Doshi](https://github.com/fenil25)
- [Jimit Gandhi](https://github.com/jimitgandhi)
- [Parth Kulkarni](https://github.com/ParthPK)

External code that was utilized in the project - 
- Some parts of `train` function in the file `train.py` was borrowed from examples in MinkowskiEngine
- Volumetric model of Poxelnet was inspired by the *MinkowskiFCNN model* provided in MinkowskiEngine Examples. Link to the example can be found [here](https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/classification_modelnet40.py).


### Individual Contributions 
- **Fenil Doshi:** The entire ```transformations.py``` file and `modelnet40_dataset.py` alongwith some parts of the `model.py` file was programmed by Fenil. 
- **Jimit Gandhi:** Jimit worked on writing the code for `create_point_cloud.py` and worked on `train.py` as well as `test.py` file. 
- **Parth Kulkarni:** Parth was responsible for writing the code for helper functions in `utils.py`. He also worked jointly with Fenil and Jimit to work on the `train.py`, `test.py` and `model.py` files. 




