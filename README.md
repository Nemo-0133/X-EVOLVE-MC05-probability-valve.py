# MC-05 Probability Valve 

MC-05 is a lightweight, asynchronous Logits interception valve designed to be mounted on the output layer (Sampler) of Large Language Models (LLMs). 

It addresses the common issues of **local optima (repetitive loops)** and **hallucination deadlocks** in smaller models (e.g., 8B parameters) by introducing a mathematical "Controlled Disturbance" via Stochastic Differential Equations (SDE).

## Core Concept: Controlled Disturbance Logic

Standard LLM sampling (like Top-K or Nucleus Sampling) relies on static mathematical probabilities. MC-05 introduces a dynamic, physics-inspired approach:

1. **Semantic & Entropy Triggers**: The valve monitors the generation stream. It only activates when it detects specific logical transition words (e.g., "however", "therefore") OR when the probability distribution is highly flat (High Entropy > 0.7), indicating model hesitation.
2. **Logits Interception**: It pauses the main inference thread, intercepts the Top-K logits, and detaches them from the main GPU graph.
3. **SDE Evolution (External)**: The logits are sent to an isolated math engine to undergo a Jump-Diffusion process ($dX = -\lambda X dt + \sigma dW$). 
4. **Zeno Collapse**: After applying extreme mathematical disturbance, the most resilient token is selected and injected back into the LLM's generation queue.

## Features

* **Zero-Blocking Architecture**: Utilizes Python's `asyncio`. The IPC (Inter-Process Communication) call to the external math engine is fully asynchronous.
* **Timeout Fast-Pass**: Built-in fault tolerance. If the external SDE engine takes longer than the defined `max_timeout` (default 0.5s), the valve auto-bypasses and returns the standard `argmax` token, ensuring TPS (Tokens Per Second) never fully collapses.
* **Semantic Whitelisting**: Conserves compute power by only triggering the SDE sandbox on crucial structural nodes of a sentence, rather than wasting resources on standard nouns or syntax.

## Integration Guide

MC-05 is designed to be injected into standard LLM inference pipelines (such as `llama.cpp` Python bindings, `vLLM`, or HuggingFace `transformers` LogitsProcessors).

```python
# Example pseudo-integration
from mc05_probability_valve import MC05_ProbabilityValve

valve = MC05_ProbabilityValve(tokenizer=my_tokenizer)

# Inside your generation loop:
# current_logits = shape(vocab_size)
# top_k_ids = shape(K)

survivor_token_id = await valve.intercept_logits_async(
    current_logits=current_logits, 
    top_k_ids=top_k_ids, 
    ns_complex_flag=True
)
