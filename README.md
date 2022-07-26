# OpenOceans 

A repository for hosting tools related to ICESat-2 bathymetry.

You've found this while its still under development! Things may not work exactly as intended while bugs get ironed out.

## Labeling Tool
ICESat-2 data is provided as individual transects of photon-counting lidar returns. Point cloud features such as ground, canopy, sea surface, and bathymetry are easily distinguished by humans, but automated methods of feature extraction are still an open area of research. 

This labeling tool can be used to select bathymetric returns in ICESat-2 data, model water surface returns, and apply depth corrections to account for water refraction. Data can be read in from a locally stored ATL03 H5 file or using API querying.

### Installation
The labeling tool is almost entirely contained within the `manual_tool.py` Python script. However, there are several  Python packages that are required to run this script. These dependencies are listed in the `environment.yaml` file. 

It's recommended to use conda or a similar package manager for configuring a development environment. Once conda is configured for your system, use the following command to set up a development environment from the `environment.yml` file.

```
    conda env create -n openoceans --file environment.yml
```

# Author

__Jonathan Markel__<br />
Graduate Research Assistant<br /> 
3D Geospatial Laboratory<br />
The University of Texas at Austin<br />
jonathanmarkel@gmail.com<br />


