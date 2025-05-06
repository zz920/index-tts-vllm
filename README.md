## 项目简介
IndexTTS-vllm

### RTF
XXX

### 新特性
XXX

## 环境准备

### 1. 创建并激活 conda 环境
```bash
conda create -n index-tts-vllm python=3.12
conda activate index-tts-vllm python
```


### 2. 安装 pytorch 2.5.1（对应 vllm 0.7.3）
```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```


### 3. 安装依赖
```bash
pip install -r requirements.txt
```


### 4. 模型下载

此为官方权重文件，下载到任意路径即可

| **HuggingFace**                                          | **ModelScope** |
|----------------------------------------------------------|----------------------------------------------------------|
| [😁IndexTTS](https://huggingface.co/IndexTeam/Index-TTS) | [IndexTTS](https://modelscope.cn/models/IndexTeam/Index-TTS) |


### 5. 模型权重转换
此操作将官方模型权重转换为 transformers 库兼容的版本，方便后续 vllm 库加载模型权重


## 推理
第一次可能会久一些，因为要 bigvgan cuda 编译