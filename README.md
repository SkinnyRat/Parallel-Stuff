## Parallel-Stuff ## 
###### Stuff that uses parallel algorithms that can be executed on GPUs or similar hardware. ###### 
--- 

### 01. Trees ### 

This tries to identify the location of trees in a satellite photo by dividing the photo into grid cells and checking how 'green' each cell is. 

First it filters the green channel by applying a HSV mask: 
![Map2](https://user-images.githubusercontent.com/13679090/39669457-b15140e0-511e-11e8-9e07-f5862c1f5962.jpg) 

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

### 02. Roofs ### 

![osm](https://user-images.githubusercontent.com/13679090/40649981-cd2a9106-6364-11e8-882b-f18b757d407a.jpg | width=400) 
![output](https://user-images.githubusercontent.com/13679090/40649982-cd68d2a4-6364-11e8-8f67-c7bfaa1fe549.jpg | width=400) 

---
