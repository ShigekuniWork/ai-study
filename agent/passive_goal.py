from langchain_core.prompts.chat import ChatPromptTemplate

from langchain_ollama.chat_models import ChatOllama
from pydantic import BaseModel, Field
from typing import cast

class Goal(BaseModel):
    description: str = Field(..., description="目標の説明")

    @property
    def text(self) -> str:
        return f"{self.description}"

class PassiveGoalCreator:
    def __init__(self, llm: ChatOllama):
        self.llm = llm

    def run(self, query: str) -> Goal:
        prompt = ChatPromptTemplate.from_template(
            "ユーザーの入力を分析し、明確で実行可能な目標を生成してください。\n"
            "要件:\n"
            "1. 目標は具体的かつ明確であり、実行可能なレベルで詳細化されている必要があります。\n"
            "2. あなたが実行可能な行動は以下の行動だけです。\n"
            "   - インターネットを利用して、目標を達成するための調査を行う。\n"
            "   - ユーザーのためのレポートを生成する。\n"
            "3. 決して2.以外の行動を取ってはいけません。\n"
            "ユーザーの入力: {query}"
        )
        chain = prompt | self.llm.with_structured_output(Goal)
        return cast(Goal, chain.invoke({"query": query}))

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="PassiveGoalCreatorを利用して目標を生成します"
    )
    parser.add_argument("--task", type=str, required=True, help="実行するタスク")
    args = parser.parse_args()

    llm = ChatOllama(model="llama3.1:latest", temperature=0.2)
    goal_creator = PassiveGoalCreator(llm=llm)
    result: Goal = goal_creator.run(query=args.task)

    print(f"{result.text}")


if __name__ == "__main__":
    main()
