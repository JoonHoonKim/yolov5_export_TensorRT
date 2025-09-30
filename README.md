# yolov5_export_TensorRT
I've added INT8 dtype code to the export.py code in the ultralytics yolov5 repository.   
---
# Setup
**1. Download TensorRT**
(Download Link)(https://developer.nvidia.com/tensorrt/download/10x)

# How To Use
**FP32**  

```python
python export.py
``` 
**FP16**  
```python
python export.py --half
```
**INT8**  
```python
python export.py --int8
```
---  
# Notice
**If you want to export INT8 dtype, Please add more calib data in calib dir at least two-hundreds**
