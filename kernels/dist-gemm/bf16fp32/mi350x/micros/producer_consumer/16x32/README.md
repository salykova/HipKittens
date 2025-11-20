# Distributed Producer-Consumer GEMM Kernel


## Start docker

```bash
 docker run -it \
    --ipc=host \
    --network=host \
    --privileged \
    --cap-add=CAP_SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    -v $(pwd):/workdir/ \
    -e USE_FASTSAFETENSOR=1 \
    -e SAFETENSORS_FAST_GPU=1 \
    -v $(pwd):/HipKittens \
    -w /HipKittens \
    rocm/7.0-preview:rocm7.0_preview_pytorch_training_mi35x_beta \
    bash
```
## Install dependacies

```terminal
apt-get install -y libopenmpi-dev openmpi-bin
```

## Build and Run

```terminal
cd kernels/dist-gemm/bf16fp32/mi350x/micros/producer_consumer/16x32/
cmake -B build
cmake --build build --parallel 16 
python3 example.py
```
