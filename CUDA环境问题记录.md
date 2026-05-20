# CUDA 环境问题记录

## 问题描述
`image_on_cuda=True` 需要完整的 CUDA Toolkit 支持，但当前环境只有 CUDA Driver（12.7），缺少运行时库和 nvcc 编译器。

## 错误信息
```
AssertionError: Can not enable cuda rendering pipeline
```
或
```
CuPy: CUDA path could not be detected. Set CUDA_PATH environment variable if CuPy fails to load.
ImportError: DLL load failed: 找不到指定的模块。
```

## 环境现状
- 系统驱动: CUDA 12.7 (RTX 4060 Laptop)
- PyTorch: 编译时使用 CUDA 11.7 (`torch.version.cuda = 11.7`)
- 缺少: CUDA Toolkit (nvcc) 和 CUDA runtime DLLs

## 解决方案
1. **安装 CUDA Toolkit 11.7** (推荐)
   - 下载: https://developer.nvidia.com/cuda-11-7-1-download-archive
   - 选择: Windows > x86_64 > local installer
   - 安装后设置环境变量: `CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7`

2. **或者降级到 cupy-cuda11x** (如果系统有 CUDA 11.x runtime)

3. **保持 image_on_cuda=False** (当前方案)
   - 分辨率仍可设为 320x180
   - 会有 buffer 过大警告但不影响训练

## 验证命令
```powershell
# 检查 CUDA 驱动版本
nvidia-smi

# 检查 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 检查 nvcc 是否可用
nvcc --version

# 检查 CUDA_PATH 环境变量
$env:CUDA_PATH
```
