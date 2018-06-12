<h2> Parallel-Stuff </h2> 
<h6> Stuff that uses parallel algorithms that can be executed on GPUs or similar hardware. </h6> 

<h6> Environment: Ubuntu 16.04 <br/> Dependencies: Cuda, OpenCV. </h6> 

--- 

### 01. Tree Locations 

This tries to identify the location of trees in a satellite photo by dividing the photo into grid cells and checking how 'green' each cell is. 

First it filters the green channel by applying a HSV mask: 
<img src="https://user-images.githubusercontent.com/13679090/39669457-b15140e0-511e-11e8-9e07-f5862c1f5962.jpg" width='600'> 
_Original satellite photo from Google Maps._<br/>
Then it checks each pixel in parallel and adds the 'green'-ness to its respective grid cell. If the sum is within a pre-defined range, that cell probably has a tree in it: 
``` 
----TTTTTTTTTT--T-------
---TTTTTT-TTT-----------
-----TTTTTTTT---T-------
-----TT-TT--------------
----T--TT--T------------
--------T---------------
T----T-TTT----TTTT------
T-T------------T-T-----T
T-----------------------
T-----------------------
T----T------------------
---TTT------------------
```

The adjustable parameters are the cell size (default 80 pixels), 'green' range (default 20-64 out of 255), and the HSV values. 

**NOTE**: This assumes the image is 1920x1080; larger images can cause it to crash if the # of rows > Cuda's block size. I should probably add a way to divide the block and grid sizes more automatically for weird-sized photos... 

--- 

### 02. Solar Roofs 

This tries to determine the total roof area available for solar panel installations in a neighbourhood, and takes into account a fire safety regulation that prescribes a walking buffer between the panels and the edge of the roof. <br/>
_Map images from OpenStreetMaps._ <br/> 
<img src="https://user-images.githubusercontent.com/13679090/40649981-cd2a9106-6364-11e8-882b-f18b757d407a.jpg" width='600'> 
<br/>Roof solar areas highlighted in orange: <br/> 
<img src="https://user-images.githubusercontent.com/13679090/40649982-cd68d2a4-6364-11e8-8f67-c7bfaa1fe549.jpg" width='600'>

A current limitation is that the algorithm is limited to only 1 roof colour, so if there are spots that are not exactly the prescribed colour, the algorithm would miss them. 

For angled roofs, the kernel would additionally need to check the gradient of the adjacent cells, that they are continuous and don't exceed regulatory angles. At the time of writing, such semantics aren't available to the OSM dataset yet. 

---
