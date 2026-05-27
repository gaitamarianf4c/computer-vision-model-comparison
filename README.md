# Computer Vision Model Comparison

A modular PyTorch project for the research topic:

**A Comparative Study of AI Models for Image Classification**

The project uses the CIFAR-10 dataset and compares multiple image classification models:

- `simple_cnn`
- `resnet18`
- `mobilenet_v3_small`
- `vit_b_16`

## Experimental Results

The models were evaluated on the CIFAR-10 dataset in a controlled CPU-only Docker environment.  
The container was limited to 2 CPU cores and 6 GB RAM, and PyTorch was configured to use 2 CPU threads.

| Model | Pretrained | Frozen Backbone | Image Size | Epochs | Batch Size | Test Accuracy | Test Loss | Training Time (s) | Total Parameters | Trainable Parameters |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Simple CNN | No | No | 32 | 5 | 64 | 72.96% | 0.7945 | 232.37 | 545,098 | 545,098 |
| ResNet-18 | Yes | Yes | 224 | 3 | 64 | 80.61% | 0.5742 | 6,666.81 | 11,181,642 | 5,130 |
| MobileNetV3 Small | Yes | Yes | 224 | 3 | 64 | 84.25% | 0.4626 | 2,726.10 | 1,528,106 | 10,250 |
| Vision Transformer B/16 | Yes | Yes | 224 | 3 | 16 | 95.45% | 0.1427 | 48,724.01 | 85,806,346 | 7,690 |

### Summary

The Simple CNN model was the fastest baseline, but achieved the lowest accuracy because it was trained from scratch. ResNet-18 improved accuracy using transfer learning, but required significantly more CPU time. MobileNetV3 Small provided the best practical balance between accuracy and training time among the convolutional models. Vision Transformer B/16 achieved the highest accuracy, but also had the highest computational cost.

### Raw CSV Results

```csv
model,model_display_name,dataset,pretrained,freeze_backbone,image_size,epochs,batch_size,learning_rate,cpu_threads,total_parameters,trainable_parameters,training_time_seconds,train_loss,train_accuracy,test_loss,test_accuracy,device,seed
simple_cnn,Simple CNN,CIFAR-10,False,False,32,5,64,0.001,2,545098,545098,232.37,0.6646,0.7676,0.7945,0.7296,cpu,42
resnet18,ResNet-18,CIFAR-10,True,True,224,3,64,0.001,2,11181642,5130,6666.81,0.5963,0.7931,0.5742,0.8061,cpu,42
mobilenet_v3_small,MobileNetV3 Small,CIFAR-10,True,True,224,3,64,0.001,2,1528106,10250,2726.1,0.5025,0.8269,0.4626,0.8425,cpu,42
vit_b_16,Vision Transformer B/16,CIFAR-10,True,True,224,3,16,0.001,2,85806346,7690,48724.01,0.1182,0.9613,0.1427,0.9545,cpu,42
```

<hr>

Docker is the recommended way to run the project because it provides a stable and reproducible environment.

## Quick start with Docker

```bash
docker compose up --build
```

Run one specific model:

```bash
docker compose run --rm app --model-name simple_cnn --epochs 5 --cpu-threads 2
docker compose run --rm app --model-name resnet18 --epochs 3 --pretrained --freeze-backbone --cpu-threads 2
docker compose run --rm app --model-name mobilenet_v3_small --epochs 3 --pretrained --freeze-backbone --cpu-threads 2
docker compose run --rm app --model-name vit_b_16 --epochs 3 --batch-size 16 --pretrained --freeze-backbone --cpu-threads 2
```

Run all models:

```bash
chmod +x scripts/run_all_models.sh
./scripts/run_all_models.sh
```

## Output

Results are appended to:

```text
results/results.csv
```


## Controlled resources

The Docker Compose file limits the container to 2 CPUs and 6 GB RAM. The application also accepts `--cpu-threads`, which configures PyTorch CPU threading. This makes local experiments more controlled, although different physical machines can still produce different absolute training times.
