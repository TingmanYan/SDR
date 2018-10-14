# FDR
Code for 'Fast Disparity Refinement with Occlusion Handling for Stereo Matching'  
## Dependency
-OpenCV 3  
-Eigen
## Usage
```
mkdir build
cd build
- on Windows:
  cmake .. -G "Visual Studio 15 2017 Win64" -T host=x64
  open and compile fdr.sln using Visual Studio 2017
- on Mac & Ubuntu:
  cmake ..
  make -j4
```
To run the demo,
- on Windows:  
  double-click demo.bat
- on Mac & Ubuntu:  
  ./demo.sh  
### 
You will obtain the same results as in our paper on Windows. Results on Mac is silghtly different due to the graph-based segmentation generates different number of superpixels on Mac and Windows.
