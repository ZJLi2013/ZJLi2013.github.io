# Unlocking 3D Generative AI & Reconstruction on AMD GPUs

## Background

The field of 3D reconstruction and generation, a crucial component of Spatial AI, has seen explosive growth recently. While NVIDIA has traditionally dominated this space, AMD GPUs are now capable of enabling the vast majority of cutting-edge 3D reconstruction and generation models.

In this post, I will demonstrate how to run these top-tier models on AMD GPUs using ROCm:

*   **[PartCrafter](https://github.com/wgsxm/PartCrafter)** (Created: 2025-06-09, ⭐2.3k)
*   **[ml-sharp](https://github.com/apple/ml-sharp)** (Created: 2025-12-12, ⭐7.2k)
*   **[shap-e](https://github.com/openai/shap-e)** (Created: 2023-04-19, ⭐12.1k)
*   **[dust3r](https://github.com/naver/dust3r)** (Created: 2024-02-21, ⭐6.9k)
*   **[Difix3D](https://github.com/nv-tlabs/Difix3D)** (Created: 2025-04-26, ⭐1.0k)
*   **[fast3r](https://github.com/facebookresearch/fast3r)** (Created: 2024-10-15, ⭐1.4k)
*   **[MapAnything](https://github.com/facebookresearch/map-anything)** (Created: 2025-09-04, ⭐2.7k)
*   **[vggt](https://github.com/facebookresearch/vggt)** (Created: 2025-03-17, ⭐12.5k)
*   **[Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3)** (Created: 2025-11-14, ⭐4.5k)
*   **[3d-gs](https://github.com/graphdeco-inria/gaussian-splatting)** (Created: 2023-07-04, ⭐20.8k)

---

## Core Components & Prerequisites

Historically, there has been a gap in Computer Vision (CV) support for AMD GPUs. However, most core dependencies for modern 3D scenarios have now been adapted.

**Note:** Most prebuilt wheels for these core libraries are currently available for the **ROCm 6.4** environment. While ROCm 7.2 is the latest, adaptation for many core libs is still in progress. Therefore, the following experiments are based on **ROCm 6.4**.

### Key Prebuilt Libraries

1.  **PyTorch (ROCm build)**
    ```sh
    pip index versions torch --index-url https://download.pytorch.org/whl/rocm6.4
    ```

2.  **PyTorch Geometric (PyG)**
    *   [ROCm PyG branch](https://github.com/Looong01/pyg-rocm-build)

3.  **xFormers**
    ```sh
    pip install -U xformers=0.0.32.post2 --index-url https://download.pytorch.org/whl/rocm6.4
    ```

4.  **gsplat**
    ```sh
    # Install gsplat following https://github.com/ROCm/gsplat/tree/release/1.0.0
    pip install gsplat --index-url=https://pypi.amd.com/simple
    ```

*Other common components like PyTorch3D and Open3D work directly in the ROCm environment and do not require special steps.*

---

## Implementation Guide

Below are the setup and execution commands for running these models in a Dockerized ROCm environment.

### 1. PartCrafter
*Interactive 3D generation and editing.*

```sh
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --shm-size=32g \
  --name part-crafter \
  -v $(pwd):/workspace -w /workspace \
  rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0

cd PartCrafter 
conda create -n partcrafter python=3.12 -y
source /opt/conda/etc/profile.d/conda.sh && conda activate partcrafter

# Remove default torch from base image
pip uninstall -y torch torchvision torchaudio

# Install ROCm compatible torch
pip3 install --pre torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4  

# Install repo dependencies
bash settings/setup.sh

# Quick start 
python3 -m scripts.inference_partcrafter_scene --image_path assets/images_scene/np6_0192a842-531c-419a-923e-28db4add8656_DiningRoom-31158.png
```

### 2. ml-sharp
*High-fidelity mesh reconstruction.*

```sh
docker run --rm -it --name ml-shapr --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --shm-size=64g -v/home/zhengjli/github:/workspace -w /workspace  rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0

cd ml-sharp
conda create -n sharp python=3.12 -y
source /opt/conda/etc/profile.d/conda.sh && conda activate sharp

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4
pip install gsplat --index-url=https://pypi.amd.com/simple
pip install -r requirements_rocm.txt 

sharp predict -i data/teaser.jpg -o data/output/ --render 
```

### 3. shap-e
*Generating 3D objects from text or images.*

```sh
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --shm-size=32g \
  --name shap-e  \
  -v $(pwd):/workspace -w /workspace \
  rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0

cd shap-e 
conda create -n shap-e  python=3.12 cmake=3.14.0 -y 
source /opt/conda/etc/profile.d/conda.sh && conda activate shap-e 

pip uninstall -y torch torchvision torchaudio
pip3 install --pre torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4  

cd shap-e
pip install -e .
pip install pyyaml
python -m shap_e.examples.sample_image_to_3d 
```

### 4. dust3r
*Dense unconstrained stereo 3D reconstruction.*

```sh
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --shm-size=32g \
  --name dust3r \
  -v $(pwd):/workspace -w /workspace \
  rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0

cd dust3r 
conda create -n dust3r python=3.12 cmake=3.14.0 -y
source /opt/conda/etc/profile.d/conda.sh && conda activate dust3r

pip uninstall -y torch torchvision torchaudio
pip3 install --pre torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4  

# Bypass existing modules update in requirements
EXCLUDE_PKGS="torch|torchvision|torchaudio"
grep -vE -i '^('"$EXCLUDE_PKGS"')([<>=!~;[:space:]]|$)' requirements.txt > requirements.tmp && mv requirements.tmp requirements.txt
python3 -m pip install -r requirements.txt 
python3 -m pip install -r requirements_optional.txt

# Build croco kernel
export PYTORCH_ROCM_ARCH="gfx942"
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../

# Download weights and run
mkdir -p checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/
python3 quick_start.py 
```

### 5. Difix3D
*3D restoration and editing.*

```sh
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --shm-size=32g \
  --name difix3d \
  -v $(pwd):/workspace -w /workspace \
  rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0

cd Difix3D 
conda create -n difix3d python=3.12 cmake=3.14.0 -y
source /opt/conda/etc/profile.d/conda.sh && conda activate difix3d

pip uninstall -y torch torchvision torchaudio
pip3 install --pre torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4  
pip install -U xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/rocm6.4

EXCLUDE_PKGS="torch|torchvision|torchaudio|xformers"
grep -vE -i '^('"$EXCLUDE_PKGS"')([<>=!~;[:space:]]|$)' requirements.txt > requirements.tmp && mv requirements.tmp requirements.txt
pip install -r requirements.txt

python src/inference_difix.py \
    --model_name "nvidia/difix_ref" \
    --input_image "assets/example_input.png" \
    --ref_image "assets/example_ref.png" \
    --prompt "remove degradation" \
    --output_dir "outputs/difix" 
```

### 6. fast3r
*Fast 3D reconstruction.*

```sh
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --shm-size=32g \
  --name fast3r \
  -v $(pwd):/workspace -w /workspace \
  rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0

cd fast3r 
conda create -n fast3r python=3.12 cmake=3.14.0 -y
source /opt/conda/etc/profile.d/conda.sh && conda activate fast3r

pip uninstall -y torch torchvision torchaudio
pip3 install --pre torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4  

EXCLUDE_PKGS="torch|torchvision|torchaudio"
grep -vE -i '^('"$EXCLUDE_PKGS"')([<>=!~;[:space:]]|$)' requirements.txt > requirements.tmp && mv requirements.tmp requirements.txt
pip install -r requirements.txt
pip install -e .
python3 -m pip install --force-reinstall --no-cache-dir "setuptools==80.9.0"

export PYTORCH_ROCM_ARCH="gfx942"
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../

python3 quick_start.py --images /workspace/dataset/colors/sub/*.jpg 
```

### 7. MapAnything
*3D mapping and scene understanding.*

```sh
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --shm-size=32g \
  --name mapanything \
  -v $(pwd):/workspace -w /workspace \
  rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0

cd map-anything 
conda create -n mapanything python=3.12 -y 
source /opt/conda/etc/profile.d/conda.sh && conda activate mapanything

pip uninstall -y torch torchvision torchaudio
pip3 install --pre torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4  

pip install -e .
python3 quick_start.py --images /workspace/dataset/colors/*.jpg 
```

### 8. vggt
*Visual Geometry Ground Truth generation.*

```sh
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --shm-size=32g \
  --name vggt \
  -v $(pwd):/workspace -w /workspace \
  rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0

cd vggt
conda create -n vggt python=3.12 -y 
source /opt/conda/etc/profile.d/conda.sh && conda activate vggt

pip3 install --pre torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4  

EXCLUDE_PKGS="torch|torchvision|torchaudio"
grep -vE -i '^('"$EXCLUDE_PKGS"')([<>=!~;[:space:]]|$)' requirements.txt > requirements.tmp && mv requirements.tmp requirements.txt
pip install -r requirements.txt

python3 quick_start.py 
```

### 9. Depth-Anything-3
*State-of-the-art monocular depth estimation.*

```sh
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --shm-size=32g \
  --name da3 \
  -v $(pwd):/workspace -w /workspace \
  rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0

cd Depth-Anything-3/
conda create -n da3 python=3.12 -y
source /opt/conda/etc/profile.d/conda.sh && conda activate da3

pip uninstall -y torch torchvision torchaudio
pip3 install --pre torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/rocm6.4  
pip install -U xformers==0.0.32.post2 --index-url https://download.pytorch.org/whl/rocm6.4
pip install gsplat --index-url=https://pypi.amd.com/simple

EXCLUDE_PKGS="torch|torchvision|torchaudio|xformers|gsplat"
grep -vE -i '^('"$EXCLUDE_PKGS"')([<>=!~;[:space:]]|$)' requirements.txt > requirements.tmp && mv requirements.tmp requirements.txt
pip install -r requirements.txt

python3 basic_gs.py 
```

### 10. 3d-gs (Gaussian Splatting)
*Real-time radiance field rendering.*

```sh
git clone -b rocm --recursive https://github.com/expenses/gaussian-splatting.git

sudo docker run --rm -it  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16G --security-opt seccomp=unconfined --security-opt apparmor=unconfined -e HIP_VISIBLE_DEVICES=6 -v /home/zhengjli/github:/workspace -v /home/zhengjli/dataset:/dataset -w /workspace   rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0

pip install --no-build-isolation submodules/simple-knn 
pip install --no-build-isolation submodules/diff-gaussian-rasterization

cd gaussian-splatting 
python3 train.py --source_path /dataset/db/playroom/ --iterations 8000 --eval 
```

---

## Future Works

Most mainstream 3D generation and reconstruction models can now be deployed on AMD GPUs. However, there are still some models with native CUDA operator dependencies that are being worked on, including:

*   [EGG-Fusion](https://github.com/panxkun/eggfusion) (Pure CUDA)
*   [Trellis 2](https://github.com/microsoft/TRELLIS.2) (Pure CUDA)
*   [ShapeR](https://github.com/facebookresearch/ShapeR) (Requires torchsparse)
*   [Wonder3D++](https://github.com/xxlong0/Wonder3D/tree/Wonder3D_Plus) (Requires nerfacc)
*   [LGM](https://github.com/3DTopia/LGM) (Requires nvdiffrast)

Stay tuned for updates as we continue to bridge the gap and enable more Spatial AI capabilities on AMD hardware!
