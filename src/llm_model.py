class LocalLLMNode:
    def __init__(self, model_path: str):
        from vllm import LLM
        self.engine = LLM(model_path,cpu_offload_gb=4, swap_space= 4,  max_seq_len_to_capture=108432) 
    def chat(self, conversation, prompt):
        return self.engine.chat(conversation, prompt)

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
        response = self.model.chat(self.conversation, prompt)
        self.conversation.append({"role": "assistant", "content": response})
        return response