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
    def __init__(self, tokenizer=None, sde_handler_coro=None):
        self.tokenizer = tokenizer
        self.sde_handler = sde_handler_coro 
        
        # [防盲區 1] 語意節點白名單 (針對宏觀草稿大幅擴充)
        self.semantic_triggers = [
            "但是", "如果", "因為", "導致", "認為", "所以", "假設", "然而", 
            "或許", "另一種可能", "相對地", "反之", "綜上所述", "不可否認", 
            "矛盾的是", "推測", "取決於", "不過"
        ]
        
        # [向下相容] 保留原有的 Token ID 映射邏輯，避免依賴 tokenizer 的舊進程報錯
        self.trigger_ids = set()
        if self.tokenizer:
            try:
                for word in self.semantic_triggers:
                    encoded = self.tokenizer.encode(word)
                    if encoded:
                        self.trigger_ids.add(encoded[-1])
            except Exception:
                pass # 若 tokenizer 不支援，安靜跳過
        
        # [機械性防護]
        self.max_timeout = 2.0            # 因應草稿層級擴展至 2.0 秒
        self.cooldown_max = 3             # 坍縮後冷卻次數，防止共振死鎖
        self.current_cooldown = 0         
        self.variance_threshold = 0.3     # 草稿差異度閥值

    async def evaluate_drafts_async(self, drafts, is_complex_task):
        """
        核心路由評估迴圈 (接收 DE-03 傳來的平行草稿)
        """
        if self.current_cooldown > 0:
            self.current_cooldown -= 1
            return self._fast_pass(drafts)

        if not is_complex_task or not isinstance(drafts, list) or len(drafts) < 2:
            return self._fast_pass(drafts)

        # 語意防禦：檢查草稿中是否包含擴充後的關鍵轉折詞
        is_crucial_node = any(
            trigger in draft 
            for draft in drafts 
            for trigger in self.semantic_triggers
        )
        if not is_crucial_node:
            return self._fast_pass(drafts)

        # 效能防禦：計算草稿間的文本差異度
        variance_score = self._calculate_draft_variance(drafts)
        
        if variance_score < self.variance_threshold:
            return self._fast_pass(drafts)

        print(f"[MC-05] 偵測到宏觀邏輯分歧 (Variance: {variance_score:.2f})，將草稿路由至沙盒...")
        
        try:
            if self.sde_handler:
                survivor_draft = await asyncio.wait_for(
                    self.sde_handler(drafts, variance_score), 
                    timeout=self.max_timeout
                )
            else:
                survivor_draft = self._fast_pass(drafts)
                
            print(f"[MC-05] 沙盒評估完成，最優解已釋放。")
            self.current_cooldown = self.cooldown_max
            return survivor_draft
            
        except asyncio.TimeoutError:
            print("[WARN] MC-05 沙盒推演超時，觸發快速通關機制！")
            return self._fast_pass(drafts)
        except Exception as e:
            print(f"[ERROR] MC-05 發生未預期錯誤: {e}，強制釋放首選草稿。")
            return self._fast_pass(drafts)

    def _fast_pass(self, item):
        """快速釋放：支援草稿列表或單一 Token 回傳"""
        if isinstance(item, list) and item:
            return item[0]
        elif isinstance(item, np.ndarray):
            return np.argmax(item)
        return item

    def _calculate_draft_variance(self, drafts):
        """計算草稿差異度 (輕量級 Jaccard 距離)"""
        try:
            sets = [set(d) for d in drafts]
            distances = []
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    intersection = len(sets[i].intersection(sets[j]))
                    union = len(sets[i].union(sets[j]))
                    similarity = intersection / union if union > 0 else 1.0
                    distances.append(1.0 - similarity)
            return sum(distances) / len(distances) if distances else 0.0
        except Exception:
            return 0.0

    # ==========================================
    # --- [向下相容區] 舊版 Token 級別處理函數 ---
    # ==========================================
    def _get_top_1_prob(self, logits):
        """[Deprecated] 計算 Top-1 的絕對機率，保留供舊版監控模組呼叫"""
        try:
            probs = np.exp(logits) / np.sum(np.exp(logits))
            return np.max(probs)
        except Exception:
            return 1.0

    def _calculate_entropy(self, logits):
        """[Deprecated] 舊版 Token 級別資訊熵計算"""
        try:
            probs = np.exp(logits) / np.sum(np.exp(logits))
            return -np.sum(probs * np.log2(probs + 1e-9))
        except Exception:
            return 0.0
