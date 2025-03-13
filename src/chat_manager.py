from typing import TypedDict, List
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI
from langgraph.graph import START, StateGraph, END
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from .vector_store import VectorStore
from .memory_manager import MemoryManager
from config.config import settings


class ChatState(TypedDict):
    question: str  # user question
    context: List[Document]  # Retrieve documents
    chat_history: List  # Dialogue History
    answer: str  # Generated answers


class ChatManager:
    def __init__(self):
        self.vector_store = VectorStore()
        self.memory_manager = MemoryManager()
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            temperature=0.7
        )
        self._init_prompt_template()
        self._init_graph()

    def _init_prompt_template(self):
        system_template = """ You are a professional knowledge base assistant. Answer users' questions by combining known information and conversation history. Only use a maximum of three sentences, and the answer should be concise and to the point. """
        human_template = """Question: {question} \nKnown information: {context} \nChat history: {chat_history} \n回答:"""

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ]

        self.prompt = ChatPromptTemplate.from_messages(messages)

    def _init_graph(self):
        workflow = StateGraph(ChatState)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        self.graph = workflow.compile()

    def _retrieve(self, state: ChatState) -> dict:
        retrieved_docs = self.vector_store.similarity_search(
            state["question"],
            k=4  # Retrieve the top 4 most relevant documents
        )
        return {"context": retrieved_docs}

    def _generate(self, state: ChatState) -> dict:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        prompt_input = {
            "context": docs_content,
            "question": state["question"],
            "chat_history": state["chat_history"]
        }
        messages = self.prompt.invoke(prompt_input)
        # get LLM response
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def get_response(self, thread_id: str, query: str) -> str:
        """Process user queries and return responses"""
        # Get chat history
        memory = self.memory_manager.get_memory(thread_id)
        chat_history = memory.chat_memory.messages if memory else []

        initial_state = {
            "question": query,
            "context": [],
            "chat_history": chat_history,
            "answer": ""
        }

        config = {"configurable": {"thread_id": thread_id}}
        response = self.graph.invoke(initial_state, config=config)

        # Update chat history
        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(response["answer"])

        return response["answer"]
