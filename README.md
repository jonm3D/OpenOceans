# OpenOceans 

A repository for hosting tools related to ICESat-2 bathymetry.

You've found this while its still under development! Things may not work exactly as intended while bugs get ironed out.

Made with ❤️ and ☕️ by:
__Jonathan Markel__<br />
Graduate Research Assistant<br /> 
3D Geospatial Laboratory<br />
The University of Texas at Austin<br />
jonathanmarkel@gmail.com<br />

## Labeling Tool
ICESat-2 data is provided as individual transects of photon-counting lidar returns. Point cloud features such as ground, canopy, sea surface, and bathymetry are easily distinguished by humans, but automated methods of feature extraction are still an open area of research. 

This labeling tool can be used to select bathymetric returns in ICESat-2 data, model water surface returns, and apply depth corrections to account for water refraction. Data can be read in from a locally stored ATL03 H5 file or using API querying.

### Installation
The labeling tool is almost entirely contained within the `manual_tool.py` Python script. However, there are several  Python packages that are required to run this script.

It's recommended to use conda or a similar package manager for configuring a development environment. Once conda is configured for your system, you can use the `environment.yml` file included in this repository and the following command to create a conda environment called 'openoceans' with the required dependencies.

```
conda env create -n openoceans --file environment.yml
```

### Running the Tool
The OpenOceans manual labeling tool is implemented as a [bokeh](https://docs.bokeh.org/en/latest/index.html) app which can be run from the terminal with a single command.

```
bokeh serve --show manual_tool.py --websocket-max-message-size=300000000
```

What is this actually doing? The command `bokeh serve --show manual_tool.py` initializes a bokeh server to run the app within the python script `manual_tool.py` and shows the app in the browser window. 
ICESat-2 ATL03 data files can also be relatively large - commonly containing hundreds of MB or several GB of photon data. However, bokeh servers are not configured to read in large amounts of local data so we increase the maximum allowed file size to 3GB with `--websocket-max-message-size=300000000`.

