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
"""import time
import numpy as np
import asyncio

class MC05_ProbabilityValve:
    def __init__(self, tokenizer, sde_handler_coro=None):
        """
        初始化機率調變閥。
        :param tokenizer: LLM 的分詞器實例，用於前瞻綁定。
        :param sde_handler_coro: (可選) 外部注入的非同步沙盒處理函數。
                                 這保護了內部架構，MC-05 不需知道沙盒的實作細節。
        """
        self.tokenizer = tokenizer
        self.sde_handler = sde_handler_coro 
        
        # [前瞻綁定] 語意錨點：只在邏輯轉折處觸發，節省算力
        self.semantic_triggers = ["但是", "如果", "因為", "導致", "認為", "所以", "假設", "然而"]
        
        # 實作 Look-ahead Binding 的簡化映射，取得轉折詞的結尾 Token ID
        self.trigger_ids = set()
        for word in self.semantic_triggers:
            encoded = self.tokenizer.encode(word)
            if encoded:
                self.trigger_ids.add(encoded[-1])

        # [機械性防護]
        self.max_timeout = 0.5            # 硬性超時底線 0.5 秒
        self.cooldown_max = 3             # 坍縮後冷卻 3 個 Token，防止共振死鎖
        self.current_cooldown = 0         # 當前冷卻計數器

    async def intercept_logits_async(self, current_logits, top_k_ids, is_complex_task):
        """
        核心攔截迴圈 (放置於 Temperature 與 Penalty 結算之後)
        """
        # 1. 迴圈防禦：檢查是否在冷卻期
        if self.current_cooldown > 0:
            self.current_cooldown -= 1
            return self._fast_pass(current_logits)

        # 2. 外部總控防禦：非複雜任務直接放行
        if not is_complex_task:
            return self._fast_pass(current_logits)

        # 3. 碎裂防禦：語意錨點命中檢查
        is_crucial_node = any(token_id in self.trigger_ids for token_id in top_k_ids)
        if not is_crucial_node:
            return self._fast_pass(current_logits)

        # 4. 效能與亂碼防禦：檢查資訊熵與 Top-1 絕對機率
        entropy = self._calculate_entropy(current_logits)
        top_1_prob = self._get_top_1_prob(current_logits)
        
        # 排除低熵(很確定) 與 徹底亂碼態(Top-1 機率過低)
        if entropy < 0.7 or top_1_prob <= 0.05:
            return self._fast_pass(current_logits)

        # ==========================================
        # 觸發條件全數滿足，啟動外部沙盒投遞
        # ==========================================
        print(f"[MC-05] 偵測到高熵邏輯轉折 (Entropy: {entropy:.2f})，攔截 Token...")
        
        try:
            if self.sde_handler:
                # 呼叫外部私有通道，並加上嚴格的硬性超時
                survivor_token_id = await asyncio.wait_for(
                    self.sde_handler(top_k_ids, current_logits), 
                    timeout=self.max_timeout
                )
            else:
                # 若無外部沙盒 (如開源使用者未實作)，則模擬快速通關
                survivor_token_id = self._fast_pass(current_logits)
                
            print(f"[MC-05] 沙盒坍縮完成，最優解 Token ID：{survivor_token_id}")
            
            # 成功坍縮，進入冷卻期
            self.current_cooldown = self.cooldown_max
            return survivor_token_id
            
        except asyncio.TimeoutError:
            print("[WARN] MC-05 沙盒演化超時，觸發快速通關機制！")
            return self._fast_pass(current_logits)
        except Exception as e:
            print(f"[ERROR] MC-05 發生未預期錯誤: {e}，強制放行。")
            return self._fast_pass(current_logits)

    def _fast_pass(self, logits):
        """快速通關：回傳機率最高的 Token"""
        return np.argmax(logits)
        
    def _get_top_1_prob(self, logits):
        """計算 Top-1 的絕對機率"""
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return np.max(probs)

    def _calculate_entropy(self, logits):
        """計算資訊熵 (Entropy)"""
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return -np.sum(probs * np.log2(probs + 1e-9))
