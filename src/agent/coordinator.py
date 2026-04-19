from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agent.retriever import GuidelineRetriever
import os

class AgentState(TypedDict):
    patient_data: dict
    risk_score: float
    guidelines: List[str]
    recommendation: str

class CareCoordinatorAgent:
    def __init__(self):
        self.retriever = GuidelineRetriever()
        self.retriever.initialize_mock_data()
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview") if os.getenv("OPENAI_API_KEY") else None

    def retrieve_guidelines(self, state: AgentState):
        risk_level = "High" if state['risk_score'] > 0.7 else "Medium"
        query = f"Interventions for {risk_level} risk patient with LeadTime {state['patient_data'].get('LeadTime', 'unknown')}"
        guidelines = self.retriever.retrieve(query)
        return {"guidelines": guidelines}

    def generate_recommendation(self, state: AgentState):
        if self.llm:
            prompt = ChatPromptTemplate.from_template("""
            You are an AI Care Coordinator. Based on the patient data and guidelines below, generate a specific, actionable intervention plan to prevent a no-show.
            
            Patient Data: {patient_data}
            Predicted No-Show Risk: {risk_score}
            
            Guidelines: {guidelines}
            
            Recommendation:
            """)
            chain = prompt | self.llm
            response = chain.invoke({
                "patient_data": state['patient_data'],
                "risk_score": state['risk_score'],
                "guidelines": state['guidelines']
            })
            return {"recommendation": response.content}
        else:
            # Fallback logic if no LLM
            rec = "Recommended Actions:\n"
            if state['risk_score'] > 0.7:
                rec += "- Schedule a priority reminder call 24 hours before.\n"
            if state['patient_data'].get('LeadTime', 0) > 10:
                rec += "- Send a mid-week check-in SMS.\n"
            rec += f"- Review guidelines: {', '.join(map(str, state['guidelines']))}"
            return {"recommendation": rec}

    def build_graph(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("retrieve", self.retrieve_guidelines)
        workflow.add_node("recommend", self.generate_recommendation)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "recommend")
        workflow.add_edge("recommend", END)
        
        return workflow.compile()

    def run(self, patient_data, risk_score):
        app = self.build_graph()
        result = app.invoke({
            "patient_data": patient_data,
            "risk_score": risk_score,
            "guidelines": [],
            "recommendation": ""
        })
        return result['recommendation']
