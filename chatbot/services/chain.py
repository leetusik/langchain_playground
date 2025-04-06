import os
from typing import Dict, List, Optional

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

# System prompt template that instructs the LLM how to respond to user questions
# It defines the response format, tone, and how to handle citations
RESPONSE_TEMPLATE = """\
당신은 요식업 창업 컨설팅 업체 "창플"의 컨설턴트로, 요식업 창업에 관한 \
모든 질문에 답변하는 역할을 맡고 있습니다.

가독성을 위해 답변에 글머리 기호를 사용하세요. 

질문과 관련된 내용이 없다면, "음, 잘 모르겠네요."라고만 말하세요. 답변을 지어내지 마세요.
"""
# Environment variables for Pinecone configuration
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]


# Pydantic model defining the structure of chat requests
class ChatRequest(BaseModel):
    question: str  # The current user question
    chat_history: Optional[List[Dict[str, str]]]  # Previous conversation history


def format_chat_history(chat_history: List[Dict[str, str]]) -> List:
    """
    Formats chat history from the API format to LangChain messages format.

    Args:
        chat_history: List of chat history entries with 'human' and 'ai' keys

    Returns:
        List: Formatted chat history as LangChain message objects
    """
    formatted_history = []

    for entry in chat_history:
        if "human" in entry:
            formatted_history.append(HumanMessage(content=entry["human"]))
        if "ai" in entry:
            formatted_history.append(AIMessage(content=entry["ai"]))

    return formatted_history


def create_chain(llm: LanguageModelLike) -> Runnable:
    """
    Creates a conversational chain using the pipe (|) operator.

    Args:
        llm: The language model for generating responses

    Returns:
        Runnable: The conversational chain
    """
    # Create the chat prompt with system instructions, chat history, and user question
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),  # Include chat history
            ("human", "{question}"),  # Current question
        ]
    )

    # Function to prepare inputs for the chain
    def prepare_inputs(inputs: dict) -> dict:
        # Extract and format chat history if it exists
        chat_history = inputs.get("chat_history", [])
        formatted_history = format_chat_history(chat_history)

        return {"chat_history": formatted_history, "question": inputs["question"]}

    # Build the chain using the pipe operator
    conversation_chain = (
        RunnableLambda(prepare_inputs) | prompt | llm | StrOutputParser()
    )

    return conversation_chain


# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

# Initialize conversation chain
answer_chain = create_chain(llm)
