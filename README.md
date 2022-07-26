# OpenOceans 

OpenOceans is a repository of tools for ICESat-2 bathymetry science.

You've found this while its still under development! Things may not work exactly as intended while bugs get ironed out.  

Made with ❤️ and ☕️ by:

__Jonathan Markel__<br />
Graduate Research Assistant<br /> 
3D Geospatial Laboratory<br />
The University of Texas at Austin<br />
jonathanmarkel@gmail.com<br />

## Labeling Tool
ICESat-2 data is provided as individual transects of photon-counting lidar returns. Point cloud features such as ground, canopy, sea surface, and bathymetry are easily distinguished by humans, but automated methods of feature extraction are still an open area of research. 

This tool can be used to 
- select bathymetric returns in ICESat-2 data
- model water surface returns
- apply depth corrections to account for water refraction

Data can be read in from a locally stored ATL03 H5 file or using API querying.

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

### Labeling Photon Data

The tool window can be broken up into several key elements. 
1. The upper left contains an assortment of buttons for input/output, point selection, refraction, and other data manipulation tools. 
2. The larger __Primary__ window is multi-purpose, depending on the tab selected. 
    - *Photon Cloud*. This tab shows the photon/elevation data for whichever file the user selects. This is also where the user will select bathymetric returns and visualize refraction corrected data before saving any output. 
    - *API QUERY*. This tab allows the user to query NASA servers by selecting a bounding box and more. 
3. Above the primary window is a __Status Update__ box for communicating important information to the user. 
4. On the bottom left is the __Preview__ window, which will display the selected ICESat-2 track over an imagery basemap. This will dynamically update with the primary window.

#### Using ATL03 H5 Files
ICESat-2 photon elevation data is typically provided by NSIDC and EarthData in the form of ATL03 H5 files. These files can be read into the labeler from local storage using the `choose file` input button in the upper left upon start up. This may take a moment for larger files, and the terminal will show an updating progress bar as each of the 6 beams are loaded. Once all data has been read, the user will be able to select a beam ('gt1r', 'gt2l', etc...) from the buttons on the top left of the window. Selecting a profile will load the corresponding photon data into the __Primary__ window.

#### Using API Querying
It can also be useful to pull ICESat-2 data from NASA servers, rather than storing it locally. This can be done using the API QUERY functionality built into the OpenOcean manual labeling tool. 

> TIP: Using the API query can take a long time depending on the size of your query, your network connection, your computers memory, and more. It is **highly** recomended to start with a relatvely small bounding box and time window (several weeks to a month) first before making larger data requests. When in doubt, only download what you need!

0. Select the *API QUERY* tab. Note that this tab contains an __Bounding Box Selection__ imagery window - this is separate from the __Preview__ window. Also, the `choose file` button has been replaced with two drop down menus to select by a profile's overpass date and by more detailed track info, including the reference ground track (RGT), cycle number, beam pair, and beam track.

1. Zoom to Area of Interest. Use the Pan, Box Zoom, or Wheel Zoom tools to the right of the __Bounding Box Selection__ window to navigate to the desired area of interest.

2. Drawing a bounding box. Select the Box Edit Tool from the tool list on the right. Hold down shift while using the mouse to draw the bounding box. This box can be redrawn as needed, but only the latest bounding box is used for the actual query.

3. Refine query parameters. Use the start and end date fields to set a time window for your query. The default options for release, surface type, and photon confidence should be sufficient for the majority of use cases. If you think you may need to change these, check out the documentation for [SlideRule](http://icesat2sliderule.org/rtd/) and the [ATL03 data dictionary](https://nsidc.org/data/atl03) technical reference for more details.

4. Click the Submit Query button to start downloading data. It is very likely that this will take at least several minutes, so please be patient! Once all the requested photon data has been downloaded, it is reformatted and the tracks are plotted in the same window as the bounding box.

5. Select an ICESat-2 profile. Hovering the mouse over the new tracks in your bounding box will tell you which date, reference ground track (RGT) and other details correspond to that specific profile. These details have been populated in the two drop down windows on the left ('Date', and 'RGT, CYCLE...'). 

