class LocalLLMNode:
    def __init__(self, model_path: str):
        from vllm import LLM, SamplingParams 
        self.sampling_params = SamplingParams(temperature=0.5, top_p=0.9)
        self.engine = LLM(model_path,cpu_offload_gb=4, swap_space= 4, gpu_memory_utilization=0.8)
    def chat(self, conversation):
        return self.engine.chat(conversation, sampling_params=self.sampling_params, use_tqdm=False)

class LLMModel:
    def __init__(self, model_path: str):
        self.model = LocalLLMNode(model_path)
        # Initialize conversation context once.
        self.conversation = [
            {"role": "system", "content": "You are a helpful assistant for answering questions based on the provided embedded documents."}
        ]
        
    # Now chat() manages the conversation internally.
    def chat(self, prompt):
        self.conversation.append({"role": "user", "content": prompt})
        outputs = self.model.chat(self.conversation)
        response = [output.outputs[0].text for output in outputs] 
        self.conversation.append({"role": "assistant", "content": response})
        return response