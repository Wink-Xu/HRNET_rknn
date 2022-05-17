## HRNet 

* test_one_image 

```
python test_one_image.py \
    --cfg config/w32_256x192_adam_lr1e-3.yaml \
    --imageFile ./output/test.jpg \
    TEST.MODEL_FILE ../model/pose_hrnet_w32_256x192.pth \
    
```

* pytorch to onnx
```
python to_onnx.py \
    --cfg config/w32_256x192_adam_lr1e-3.yaml \
    --output ./output/pose_hrnet_w32_256x192.onnx \
    TEST.MODEL_FILE ../model/pose_hrnet_w32_256x192.pth \
```

* onnx to rknn and check the result
rknn-toolkit
https://github.com/rockchip-linux/rknn-toolkit  
download the rknn-tookit docker images

```
1. prepare the quant data in dataset.txt
2. cd code/rknn/transModel
   python to_rknn.py --transModel
```

* inference rknn using C API
https://github.com/rockchip-linux/rknpu/tree/master/rknn/doc
need to configure the compile environment follow the doc, and run the code on the device 
```
cd code/rknn/inference/rknn_hrnet
./build.sh 
tar the "install" directory, move to the device
./rknn_hrnet ./model/pose_hrnet_w32_256x192.rknn ./model/test1.jpg
```

