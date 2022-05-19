## build

modify `GCC_COMPILER` on `build.sh` for target platform, then execute

```
./build.sh
```

## install

connect device and push build output into `/userdata`

```
adb push install/rknn_hrnet /userdata/
```

## run

```
adb shell
cd /userdata/rknn_hrnet/
```

- rv1126
```
./rknn_hrnet model/pose_hrnet_w32_256x192.rknn model/test.jpg
```
