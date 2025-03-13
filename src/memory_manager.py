from langchain.memory import ConversationBufferMemory
from typing import Dict

class MemoryManager:
    def __init__(self):
        self.sessions: Dict[str, ConversationBufferMemory] = {}
    
    def get_memory(self, thread_id: str) -> ConversationBufferMemory:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
        return self.sessions[thread_id]
    
    def clear_memory(self, thread_id: str):
        if thread_id in self.sessions:
            del self.sessions[thread_id]