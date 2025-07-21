import pprint

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_ollama.chat_models import ChatOllama

model = ChatOllama(model="llama3.1:latest", temperature=0.2)
output_parser = StrOutputParser()

# White Hat - Facts and Information
white_hat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a factual AI assistant. Focus only on facts, data, and objective information. Avoid opinions and emotions.",
        ),
        ("user", "{input}"),
    ]
)
white_hat_chain = white_hat_prompt | model | output_parser

# Red Hat - Emotions and Feelings
red_hat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an emotional AI assistant. Express feelings, intuitions, and gut reactions about the topic. Focus on emotional responses.",
        ),
        ("user", "{input}"),
    ]
)
red_hat_chain = red_hat_prompt | model | output_parser

# Black Hat - Critical and Cautious
black_hat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a critical AI assistant. Focus on potential problems, risks, and weaknesses. Be cautious and highlight what could go wrong.",
        ),
        ("user", "{input}"),
    ]
)
black_hat_chain = black_hat_prompt | model | output_parser

# Yellow Hat - Positive and Optimistic
yellow_hat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an optimistic AI assistant. Focus on benefits, opportunities, and positive aspects. Highlight what could work well.",
        ),
        ("user", "{input}"),
    ]
)
yellow_hat_chain = yellow_hat_prompt | model | output_parser

# Green Hat - Creative and Alternative
green_hat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a creative AI assistant. Generate new ideas, alternatives, and innovative solutions. Think outside the box.",
        ),
        ("user", "{input}"),
    ]
)
green_hat_chain = green_hat_prompt | model | output_parser

# Blue Hat - Process and Control
blue_hat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a process-focused AI assistant. Think about thinking itself, organize ideas, and provide structure and control to the discussion.",
        ),
        ("user", "{input}"),
    ]
)
blue_hat_chain = blue_hat_prompt | model | output_parser

result_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a result AI assistant. Combine the results of the previous AI assistants. Highlight the most important information and provide a summary.",
        ),
    ]
)


parallel_chain = (
    RunnableParallel(
        {
            "white_hat": white_hat_chain,
            "red_hat": red_hat_chain,
            "black_hat": black_hat_chain,
            "yellow_hat": yellow_hat_chain,
            "green_hat": green_hat_chain,
            "blue_hat": blue_hat_chain,
        }
    )
    | result_prompt
    | model
    | output_parser
)

output = parallel_chain.invoke({"input": "RAGの今後について"})
pprint.pprint(output)