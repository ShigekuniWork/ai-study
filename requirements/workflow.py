from typing import Any, Optional

from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from requirements.document_generator import RequirementsDocumentGenerator
from requirements.generator import PersonaGenerator
from requirements.information_evaluator import InformationEvaluator
from requirements.interview_conductor import InterviewConductor
from requirements.state import InterviewState


class DocumentationAgent:
    def __init__(self, llm: ChatOllama, k: Optional[int] = None):
        self.persona_generator = PersonaGenerator(llm=llm, k=k)
        self.interview_conductor = InterviewConductor(llm=llm)
        self.information_evaluator = InformationEvaluator(llm=llm)
        self.requirements_generator = RequirementsDocumentGenerator(llm=llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        workflow = StateGraph(InterviewState)
        workflow.add_node("generate_personas", self._generate_personas)
        workflow.add_node("conduct_interviews", self._conduct_interviews)
        workflow.add_node("evaluate_information", self._evaluate_information)
        workflow.add_node("generate_requirements", self._generate_requirements)
        workflow.set_entry_point("generate_personas")
        workflow.add_edge("generate_personas", "conduct_interviews")
        workflow.add_edge("conduct_interviews", "evaluate_information")
        workflow.add_conditional_edges(
            "evaluate_information",
            lambda s: not s.is_information_sufficient and s.iteration < 5,
            {True: "generate_personas", False: "generate_requirements"},
        )
        workflow.add_edge("generate_requirements", END)
        return workflow.compile()

    def _generate_personas(self, state: InterviewState) -> dict[str, Any]:
        new_personas = self.persona_generator.run(state.user_request)
        return {"personas": new_personas.personas, "iteration": state.iteration + 1}

    def _conduct_interviews(self, state: InterviewState) -> dict[str, Any]:
        interviews = self.interview_conductor.run(
            state.user_request, state.personas[-5:]
        )
        return {"interviews": interviews.interviews}

    def _evaluate_information(self, state: InterviewState) -> dict[str, Any]:
        evaluation = self.information_evaluator.run(
            state.user_request, state.interviews
        )
        return {
            "is_information_sufficient": evaluation.is_sufficient,
            "evaluation_reason": evaluation.reason,
        }

    def _generate_requirements(self, state: InterviewState) -> dict[str, Any]:
        doc = self.requirements_generator.run(state.user_request, state.interviews)
        return {"requirements_doc": doc}

    def run(self, user_request: str) -> str:
        initial_state = InterviewState(user_request=user_request)
        final_state = self.graph.invoke(initial_state)
        return final_state["requirements_doc"]

    def stream_final_output(self, user_request: str):
        # 途中ステップは同期で進める
        state = InterviewState(user_request=user_request)
        state.personas = self.persona_generator.run(user_request).personas
        state.interviews = self.interview_conductor.run(
            user_request, state.personas
        ).interviews

        # 評価省略（必要なら入れてもOK）

        # ストリーミングで要件定義書生成
        for chunk in self.requirements_generator.stream(user_request, state.interviews):
            yield chunk
            print(chunk)


if __name__ == "__main__":
    llm = ChatOllama(model="llama3.1:latest", temperature=0.2)
    agent = DocumentationAgent(llm=llm)

    print("=== Simple Run ===")
    print(agent.run("DB that can directly receive open telemetry"))

    print("\n=== Streaming ===")
    for chunk in agent.stream_final_output(
        "DB that can directly receive open telemetry"
    ):
        print(chunk, end="", flush=True)
