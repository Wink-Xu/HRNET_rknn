## HRNet 

* 单张图片测试 

```
python test_one_image.py \
    --cfg config/w32_256x192_adam_lr1e-3.yaml \
    --imageFile ./output/test.jpg \
    TEST.MODEL_FILE ../model/pose_hrnet_w32_256x192.pth \
    
```

* 转化成onnx
```
python to_onnx.py \
    --cfg config/w32_256x192_adam_lr1e-3.yaml \
    --output ./output/pose_hrnet_w32_256x192.onnx \
    TEST.MODEL_FILE ../model/pose_hrnet_w32_256x192.pth \
```

onnx转化成rknn


