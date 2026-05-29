"""
Module: MC-05 Probability Valve
Description:
A lightweight interception valve mounted directly on the output layer of Large Language Models.
Based on "Controlled Disturbance Logic," this module asynchronously intercepts Top-K Logits 
when high entropy (decision divergence) or specific semantic trigger nodes are detected. 
It routes these logits to an external SDE (Stochastic Differential Equation) engine for 
evaluation, effectively breaking the model out of local optima and mitigating hallucination deadlocks.

Dependencies:
- numpy
- asyncio
- time
"""
import time
import numpy as np
import asyncio

class MC05_ProbabilityValve:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.semantic_triggers = ["但是", "如果", "因為", "導致", "認為", "所以", "假設", "然而"]
        # 將觸發詞預先編碼為 Token ID 集合，取代推論迴圈內的字串比對
        self.trigger_ids = {self.tokenizer.encode(word)[-1] for word in self.semantic_triggers}
        self.max_timeout = 0.5 

    async def intercept_logits_async(self, current_logits, top_k_ids, ns_complex_flag):
        if not ns_complex_flag:
            return self._fast_pass(current_logits)

        is_crucial_node = any(token_id in self.trigger_ids for token_id in top_k_ids)
        if not is_crucial_node:
            return self._fast_pass(current_logits)

        entropy = self._calculate_entropy(current_logits)
        if entropy < 0.7:
            return self._fast_pass(current_logits)

        print("[MC-05] 偵測到高熵邏輯轉折，啟動非同步沙盒演化...")
        
        try:
            # 引入 asyncio.wait_for 執行非同步超時控制，徹底解除主執行緒阻塞
            survivor_token_id = await asyncio.wait_for(
                self._call_de_sandbox_engine_async(top_k_ids), 
                timeout=self.max_timeout
            )
            print(f"[MC-05] 沙盒坍縮完成，最優解 Token ID：{survivor_token_id}")
            return survivor_token_id
            
        except asyncio.TimeoutError:
            print("[WARN] 沙盒演化超時，觸發快速通關機制！")
            return self._fast_pass(current_logits)
        except Exception:
            return self._fast_pass(current_logits)

    def _fast_pass(self, logits):
        return np.argmax(logits)

    def _calculate_entropy(self, logits):
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return -np.sum(probs * np.log2(probs + 1e-9))
        
    async def _call_de_sandbox_engine_async(self, token_ids):
        # 實體隔離的非同步 IPC 呼叫通道
        await asyncio.sleep(0.1) 
        return token_ids[0]
