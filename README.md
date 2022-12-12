# Generate yolov7 format
    python yolo_gen.py

# Generate augmentation data
    cd Data_Augmentation
    python main.py

# Generate augmentation & rotation data
    git clone https://github.com/whynotw/rotational-data-augmentation-yolo.git
    cd rotational-data-augmentation-yolo.git
    python rotation.py ../dataset/Augment -o ../dataset/RotAug_train

# Generate multi-scale cropping data
    cd CropScale
    python CropScale.py
    python random_split.py
    
# Train
    git clone https://github.com/WongKinYiu/yolov7.git
    cd yolov7
    wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt
    
    # method 1
    python train.py --workers 8 --device 0 --batch-size 8 --data ../hyp/method1/drone-Rot_Aug.yaml --epochs 10 --img 1280 1280 --cfg cfg/training/yolov7x.yaml --name yolov7x-Drone_RotAug --weight ./yolov7x.pt  --hyp ../hyp/method1/hyp.scratch.p5-Rot_Aug.yaml

    python train.py --workers 8 --device 0 --batch-size 8 --data ../hyp/method1/drone-Aug.yaml --epochs 10 --img 1280 1280 --cfg cfg/training/yolov7x.yaml --name yolov7x-Drone_Aug --weight ./runs/train/yolov7x-Drone_RotAug/weights/best.pt --hyp ../hyp/method1/hyp.scratch.p5-Aug.yaml

    python train.py --workers 8 --device 0 --batch-size 8 --data ../hyp/method1/drone-fine_tune.yaml --epochs 25 --img 1280 1280 --cfg cfg/training/yolov7x.yaml --name yolov7x-Drone_1280finetune --weight ./runs/train/yolov7x-Drone_Aug/weights/best.pt --hyp ../hyp/method1/hyp.scratch.p5-1280fine_tune.yaml

    python train.py --workers 8 --device 0 --batch-size 8 --data ../hyp/method1/drone-fine_tune.yaml --epochs 100 --img 1920 1920 --cfg cfg/training/yolov7x.yaml --name yolov7x-Drone_1920finetune --weight ./runs/train/yolov7x-Drone_1280finetune/weights/best.pt --hyp ../hyp/method1/hyp.scratch.p5-1920fine_tune.yaml
    
    # method 2
    python train.py --workers 8 --device 0 --batch-size 8 --data ../hyp/method2/drone-Crop.yaml --epochs 100 --img 1280 1280 --cfg cfg/training/yolov7x.yaml --name yolov7x-Drone_Crop --weight ./yolov7x.pt  --hyp ../hyp/method2/hyp.scratch.p5-Crop.yaml

    python train.py --workers 8 --device 0 --batch-size 8 --data ../hyp/method2/drone-Crop_withHyper_fine_tune.yaml --epochs 50 --img 1920 1920 --cfg cfg/training/yolov7x.yaml --name yolov7x-Drone_Crop_withHyper_finetune --weight ./runs/train/yolov7x-Drone_Crop/weights/best.pt  --hyp ../hyp/method2/hyp.scratch.p5-withHyper_fine_tune.yaml

    # method 3
    python train.py --workers 8 --device 0 --batch-size 8 --data ../hyp/method3/drone-Crop.yaml --epochs 100 --img 1280 1280 --cfg cfg/training/yolov7x.yaml --name yolov7x-Drone_Crop --weight ./yolov7x.pt  --hyp ../hyp/method3/hyp.scratch.p5-Crop.yaml

    python train.py --workers 8 --device 0 --batch-size 8 --data ../hyp/method3/drone-Crop_woHyper_fine_tune.yaml --epochs 50 --img 1920 1920 --cfg cfg/training/yolov7x.yaml --name yolov7x-Drone_Crop_woHyper_finetune --weight ./runs/train/yolov7x-Drone_Crop/weights/best.pt  --hyp ../hyp/method3/hyp.scratch.p5-woHyper_fine_tune.yaml

# Validation
    python test.py --weights {Weights_Path} --conf 0.25 --iou 0.5 --img-size 2560 --data ../hyp/drone-val.yaml --name {Val_Path} --save-txt --batch-size 16 --no-trace
    cp runs/test/{Val_Path} ../TIoU/predict
    cd ../TIoU
    python TIoU.py

# Detect
    python detect.py --weights {Weights_Path} --conf 0.25 --img-size 2560 --source ../dataset/test --name {Detect_Output} --save-txt --nosave --no-trace
    cd ../
    python label2ans.py {Detect_Output}