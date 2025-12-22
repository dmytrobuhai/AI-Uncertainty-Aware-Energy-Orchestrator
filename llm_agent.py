import os
from pathlib import Path
from typing import TypedDict, Annotated, List

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langgraph.graph import StateGraph, END

# --- Завантаження конфігурації ---
def load_config():
    config_path = Path(".config")
    if config_path.exists():
        for line in config_path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

load_config()

# --- Визначення стану агента ---
class AgentState(TypedDict):
    region: str
    data: dict
    weather: dict
    balancing: List[dict]
    context: str  # Отримані знання з RAG
    explanation: str # Фінальна відповідь

class EnergyGraphAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = self._prepare_rag()
        self.workflow = self._build_graph()

    def _prepare_rag(self):
        """Створює векторну базу знань з текстових інструкцій"""
        kb_path = Path("data/knowledge_base.txt")
        if not kb_path.exists():
            # Базовий контент, якщо файл відсутній
            content = "Черга 1: -50МВт. Черга 2: -100МВт. Аварійні: > -200МВт. При вітрі > 15м/с ризик обриву +20%."
            kb_path.parent.mkdir(exist_ok=True)
            kb_path.write_text(content, encoding="utf-8")
        
        text = kb_path.read_text(encoding="utf-8")
        text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
        return FAISS.from_documents(docs, self.embeddings)

    # --- Вузли графа ---
    def retrieve_knowledge(self, state: AgentState):
        """Вузол 1: Пошук релевантних правил у базі знань"""
        query = f"Режим роботи при дефіциті {state['data']['schedule'][0]['deficit']} МВт"
        docs = self.vector_store.similarity_search(query, k=2)
        context = "\n".join([d.page_content for d in docs])
        return {"context": context}

    def analyze_and_explain(self, state: AgentState):
        """Вузол 2: Формування обґрунтування на основі даних та контексту"""
        prompt = f"""
        Ти - старший диспетчер енергосистеми України.
        Об'єкт: {state['region']}
        
        ПОТОЧНІ ДАНІ:
        - Прогноз дефіциту: {[s['deficit'] for s in state['data']['schedule']]} МВт
        - Стратегічний ризик: {state['data']['strategic_risk_score']:.2%}
        - Погода: {state['weather']['temperature'][0]}°C, вітер {state['weather']['wind_speed'][0]} км/год
        - Міжрегіональна допомога: {state['balancing']}
        
        ПРАВИЛА ТА РЕГЛАМЕНТИ (RAG):
        {state['context']}
        
        ЗАВДАННЯ:
        Дай коротке технічне обґрунтування (3-4 речення). 
        Чому обрано такий режим (стабільний/черги/аварійний)? 
        Як впливає погода або ризики на результат? Поясни логіку балансування, якщо воно є.
        Мова: українська.
        """
        response = self.llm.invoke(prompt)
        return {"explanation": response.content}

    def _build_graph(self):
        """Створення структури LangGraph"""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("retrieve", self.retrieve_knowledge)
        workflow.add_node("explain", self.analyze_and_explain)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "explain")
        workflow.add_edge("explain", END)
        
        return workflow.compile()

    def run(self, region, data, weather, balancing):
        inputs = {
            "region": region,
            "data": data,
            "weather": weather,
            "balancing": balancing
        }
        result = self.workflow.invoke(inputs)
        return result["explanation"]