import random
import patch_vllm  # ⚠️ Monkey Patch, do not delete this line

import time
import uuid
import torch
from vllm import AsyncLLMEngine, SamplingParams, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs
import asyncio
import os
# os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
total_req_num = 0

async def main():
    model_path = "/path/to/IndexTeam/Index-TTS"
    
    engine_args = AsyncEngineArgs(
        model=model_path,
        tensor_parallel_size=1,
        dtype="auto",
        gpu_memory_utilization=0.25,
        # enable_prefix_caching=False,
    )
    
    llm = AsyncLLMEngine.from_engine_args(engine_args)
    
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.8,
        top_k=30,
        repetition_penalty=10.0,
        max_tokens=768,
    )

    req_num = 16  # 并发请求量

    async def continuous_request():
        """持续发送请求的循环任务"""
        while True:
            # 动态生成新的提示
            prompt_length = random.randint(32, 128)
            prompt = list(range(prompt_length))
            prefix_embeds = torch.rand(1, prompt_length, 1024)
            token_prompt = TokensPrompt(
                prompt_token_ids=prompt,
                multi_modal_data={"image": prefix_embeds}
            )
            
            # 请求指标记录
            ttft = -1
            stt = time.perf_counter()
            
            # 生成唯一请求ID
            request_id = str(uuid.uuid4())
            
            # 发送请求并处理响应
            output_generator = llm.generate(
                token_prompt,
                sampling_params=sampling_params,
                request_id=request_id
            )
            
            async for output in output_generator:
                if ttft == -1:
                    ttft = time.perf_counter() - stt
            
            # 计算性能指标
            total_time = time.perf_counter() - stt
            latency = total_time - ttft if ttft != -1 else 0
            tokenpersec = len(output.outputs[0].token_ids) / latency if latency > 0 else 0
            
            global total_req_num
            total_req_num += 1
            # with open("/data/jcxy/hhy/workspace/index-tts-vllm/test.txt", "w") as f:
            #     f.write(str(total_req_num))
            print(f"Request {request_id[:8]}... | "
                  f"TTFT: {ttft:.2f}s | "
                  f"Total: {total_time:.2f}s | "
                  f"Speed: {tokenpersec:.1f} tok/s")

    # 创建并运行持续请求任务
    tasks = [asyncio.create_task(continuous_request()) for _ in range(req_num)]
    
    # 保持任务持续运行
    await asyncio.gather(*tasks)

asyncio.run(main())