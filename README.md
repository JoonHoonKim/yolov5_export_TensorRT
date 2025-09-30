# yolov5_export_TensorRT
I've added INT8 dtype code to the export.py code in the ultralytics yolov5 repository. 

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
