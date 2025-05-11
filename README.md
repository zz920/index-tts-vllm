<div align="center">

# IndexTTS-vLLM
</div>

## 项目简介
该项目在 [index-tts](https://github.com/index-tts/index-tts) 的基础上使用 vllm 库重新实现了 gpt 模型的推理，加速了 index-tts 的推理过程。

推理速度在单卡 RTX 4090 上的提升为：
- 单个请求的 RTF (Real-Time Factor)：≈0.3 -> ≈0.1
- 单个请求的 gpt 模型 decode 速度：≈90 token / s -> ≈280 token / s
- 并发量：gpu_memory_utilization设置为0.5（约12GB显存）的情况下，vllm 显示 `Maximum concurrency for 608 tokens per request: 237.18x`，两百多并发，man！当然考虑 TTFT 以及其他推理成本（bigvgan 等）保守估计 20 左右的并发应该无压力（没实测过，手动狗头）

## 新特性
- 支持多角色音频混合：可以传入多个参考音频，TTS 输出的角色声线为多个参考音频的混合版本（输入多个参考音频会导致输出的角色声线不稳定，可以抽卡抽到满意的声线再作为参考音频）

## 使用步骤

### 1. git 本项目
```bash
git clone https://github.com/Ksuriuri/index-tts-vllm.git
cd index-tts-vllm
```


### 2. 创建并激活 conda 环境
```bash
conda create -n index-tts-vllm python=3.12
conda activate index-tts-vllm
```


### 3. 安装 pytorch 2.5.1（对应 vllm 0.7.3）
```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```


### 4. 安装依赖
```bash
pip install -r requirements.txt
```


### 5. 下载模型权重

此为官方权重文件，下载到本地任意路径即可

| **HuggingFace**                                          | **ModelScope** |
|----------------------------------------------------------|----------------------------------------------------------|
| [😁IndexTTS](https://huggingface.co/IndexTeam/Index-TTS) | [IndexTTS](https://modelscope.cn/models/IndexTeam/Index-TTS) |

### 6. 模型权重转换
将 `convert_hf_format.sh` 中的 `MODEL_DIR` 修改为模型权重下载路径，然后运行：

```bash
bash convert_hf_format.sh
```

此操作会将官方的模型权重转换为 transformers 库兼容的版本，保存在模型权重路径下的 `vllm` 文件夹中，方便后续 vllm 库加载模型权重

### 7. webui 启动！
将 `webui.py` 中的 `model_dir` 修改为模型权重下载路径，然后运行：

```bash
python webui.py
```
第一次启动可能会久一些，因为要对 bigvgan 进行 cuda 核编译
