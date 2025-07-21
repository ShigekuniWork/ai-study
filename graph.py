import operator
from typing import Annotated

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END


class State(BaseModel):
    query: str = Field(default="", description="Question from user")
    current_role: str = Field(default="", description="Current role of the agent")
    messages: Annotated[list[str], operator.add] = Field(
        default=[], description="List of messages in the conversation"
    )
    current_judge: bool = Field(
        default=False, description="Whether the current judge is correct"
    )
    judgement_reason: str = Field(
        default="", description="Reason for the judgement of the current judge"
    )


model = ChatOllama(model="llama3.1:latest", temperature=0.2)


def role_selector(state: State) -> State:
    """ユーザーのクエリから適切な役割を選択"""
    prompt = ChatPromptTemplate.from_template(
        """
        以下の質問に最も適した役割を選んでください:
        
        質問: {query}
        
        選択肢:
        - technical: 技術的な質問
        - business: ビジネス関連の質問  
        - creative: 創作関連の質問
        - general: 一般的な質問
        
        役割のみ答えてください。
        """
    )
    
    chain = prompt | model | StrOutputParser()
    role = chain.invoke({"query": state.query}).strip()
    
    return State(
        query=state.query,
        current_role=role,
        messages=state.messages + [f"役割を選択しました: {role}"],
        current_judge=state.current_judge,
        judgement_reason=state.judgement_reason
    )


def answer_generator(state: State) -> State:
    """選択された役割に基づいて回答を生成"""
    role_prompts = {
        "technical": "あなたは技術エキスパートです。専門用語を使って詳しく説明してください。",
        "business": "あなたはビジネスコンサルタントです。実用的で具体的なアドバイスをしてください。",
        "creative": "あなたは創作のプロです。想像力豊かで面白いアイデアを提供してください。",
        "general": "あなたは親しみやすいアシスタントです。分かりやすく説明してください。"
    }
    
    role_prompt = role_prompts.get(state.current_role, role_prompts["general"])
    
    prompt = ChatPromptTemplate.from_template(
        f"""
        {role_prompt}
        
        質問: {{query}}
        
        回答してください。
        """
    )
    
    chain = prompt | model | StrOutputParser()
    answer = chain.invoke({"query": state.query})
    
    return State(
        query=state.query,
        current_role=state.current_role,
        messages=state.messages + [f"回答: {answer}"],
        current_judge=state.current_judge,
        judgement_reason=state.judgement_reason
    )


def judge_answer(state: State) -> State:
    """生成された回答が適切かどうか判定"""
    last_answer = state.messages[-1] if state.messages else ""
    
    prompt = ChatPromptTemplate.from_template(
        """
        以下の質問と回答を評価してください:
        
        質問: {query}
        役割: {role}
        回答: {answer}
        
        この回答は質問に適切に答えていますか？
        「はい」または「いいえ」で答え、その理由も説明してください。
        
        フォーマット:
        判定: はい/いいえ
        理由: 理由を説明
        """
    )
    
    chain = prompt | model | StrOutputParser()
    judgment = chain.invoke({
        "query": state.query,
        "role": state.current_role,
        "answer": last_answer
    })
    
    is_good = "はい" in judgment
    
    return State(
        query=state.query,
        current_role=state.current_role,
        messages=state.messages + [f"判定結果: {judgment}"],
        current_judge=is_good,
        judgement_reason=judgment
    )


def should_continue(state: State) -> str:
    """判定結果に基づいて次のステップを決定"""
    if state.current_judge:
        return "end"
    else:
        return "retry"


def retry_with_different_role(state: State) -> State:
    """異なる役割で再試行"""
    roles = ["technical", "business", "creative", "general"]
    current_role = state.current_role
    
    # 現在の役割以外をランダム選択
    available_roles = [r for r in roles if r != current_role]
    new_role = available_roles[0] if available_roles else "general"
    
    return State(
        query=state.query,
        current_role=new_role,
        messages=state.messages + [f"別の役割で再試行: {new_role}"],
        current_judge=False,
        judgement_reason=""
    )


# グラフの構築
workflow = StateGraph(State)

# ノードを追加
workflow.add_node("role_selector", role_selector)
workflow.add_node("answer_generator", answer_generator)
workflow.add_node("judge_answer", judge_answer)
workflow.add_node("retry", retry_with_different_role)

# エッジを追加
workflow.set_entry_point("role_selector")
workflow.add_edge("role_selector", "answer_generator")
workflow.add_edge("answer_generator", "judge_answer")
workflow.add_conditional_edges(
    "judge_answer",
    should_continue,
    {
        "end": END,
        "retry": "retry"
    }
)
workflow.add_edge("retry", "answer_generator")

# グラフをコンパイル
app = workflow.compile()

# 実行例
if __name__ == "__main__":
    initial_state = State(query="Pythonでファイルを読み込む方法を教えて")
    
    for output in app.stream(initial_state):
        for key, value in output.items():
            print(f"ノード '{key}':")
            # LangGraphはdict形式で返すので辞書アクセス
            print(f"  現在の役割: {value.get('current_role', '')}")
            print(f"  判定: {value.get('current_judge', False)}")
            print(f"  メッセージ数: {len(value.get('messages', []))}")
            messages = value.get('messages', [])
            if messages:
                print(f"  最新メッセージ: {messages[-1][:100]}...")
            print("---")