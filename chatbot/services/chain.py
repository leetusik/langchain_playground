import os
from operator import itemgetter
from typing import Dict, List, Optional, Sequence

from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI
from pinecone import Pinecone

from chatbot.services.ingest_4000 import get_embeddings_model

# System prompt template that instructs the LLM how to respond to user questions
# It defines the response format, tone, and how to handle citations
RESPONSE_TEMPLATE = """\
당신은 요식업 창업 전문가이자 컨설턴트로, 요식업 창업에 관한 \
모든 질문에 답변하는 역할을 맡고 있습니다.

제공된 검색 결과(URL 및 내용)만을 기반으로 주어진 질문에 대해 400 단어 이하의 포괄적이고 \
유익한 답변을 생성하세요. 반드시 제공된 검색 결과의 정보만 사용해야 합니다. 검색 결과와 \
동일한 어투를 사용하세요. 검색 결과를 결합하여 일관된 답변을 만드세요. 글을 반복하지 마세요. \
[${{number}}] 표기법을 사용하여 검색 결과를 인용하세요. 질문에 정확하게 답변하는 가장 \
관련성 높은 결과들만 인용하세요. 이러한 인용을 참조하는 문장이나 단락의 끝에 배치하고, \
모두 끝에 모아 놓지 마세요. 같은 이름 내에서 다른 엔티티를 참조하는 다른 결과가 있다면, \
각 엔티티에 대해 별도의 답변을 작성하세요.

가독성을 위해 답변에 글머리 기호를 사용하세요. 인용은 모두 끝에 모아 놓지 말고 적용되는 부분에 배치하세요.

맥락에서 질문과 관련된 내용이 없다면, "음, 잘 모르겠네요."라고만 말하세요. 답변을 지어내지 마세요.

다음 `context` HTML 블록 사이의 모든 것은 벡터스토어에서 검색된 것이며, 사용자와의 대화의 일부가 아닙니다.

<context>
    {context} 
<context/>

기억하세요: 맥락 내에 관련 정보가 없다면, "음, 잘 모르겠네요."라고만 말하세요. 답변을 지어내지 마세요. \
앞의 'context' HTML 블록 사이의 모든 것은 벡터스토어에서 검색된 것이며, 사용자와의 대화의 일부가 아닙니다.\
"""

# Template for rephrasing follow-up questions based on chat history
# Used to convert follow-up questions into standalone questions
REPHRASE_TEMPLATE = """\
다음 대화와 후속 질문을 바탕으로, 후속 질문을 독립적인 질문으로 바꿔주세요.

대화 기록:
{chat_history}
후속 입력: {question}
독립적인 질문:"""

# Environment variables for Pinecone configuration
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]
PINECONE_INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]


# Pydantic model defining the structure of chat requests
class ChatRequest(BaseModel):
    question: str  # The current user question
    chat_history: Optional[List[Dict[str, str]]]  # Previous conversation history


def get_retriever() -> BaseRetriever:
    """
    Creates and returns a retriever connected to the Pinecone vector database.

    The retriever is responsible for finding relevant documents based on the user's query.
    It uses the text-embedding-3-small model to convert queries to vectors.

    Returns:
        BaseRetriever: A retriever that searches Pinecone for relevant documents
    """
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Get embeddings model from ingest.py
    embedding = get_embeddings_model()

    # Create Langchain Pinecone vectorstore connected to our existing index
    # This doesn't create a new index, just connects to an existing one
    vectorstore = LangchainPinecone.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedding,
        text_key="text",  # Field name where document text is stored
    )

    # Return as retriever with k=3 (retrieve 3 most relevant chunks)
    # K=3 is a good balance for Korean text, providing enough context without too much noise
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def create_retriever_chain(
    llm: LanguageModelLike, retriever: BaseRetriever
) -> Runnable:
    """
    Creates a chain that handles both direct questions and follow-up questions.

    For follow-up questions, it uses chat history to rephrase the question
    before retrieving documents. For direct questions, it retrieves immediately.

    Args:
        llm: The language model for rephrasing questions
        retriever: The retriever for finding relevant documents

    Returns:
        Runnable: A chain that handles question processing and retrieval
    """
    # Create prompt for converting follow-up questions to standalone questions
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)

    # Chain for rephrasing questions based on chat history
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )

    # Chain that takes the rephrased question and retrieves relevant documents
    conversation_chain = condense_question_chain | retriever

    # Branch logic to handle different types of questions
    return RunnableBranch(
        # If chat history exists, use it to rephrase the question first
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        # If no chat history, retrieve documents directly using the question
        (
            RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            )
            | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


def format_docs(docs: Sequence[Document]) -> str:
    """
    Formats retrieved documents into a structured string for the LLM.

    Each document includes metadata (title, URL) and content with a unique ID.
    This structured format helps the LLM understand and cite documents correctly.

    Args:
        docs: List of retrieved documents

    Returns:
        str: Formatted document string
    """
    formatted_docs = []
    for i, doc in enumerate(docs):
        # Format each document with metadata and content
        # The ID allows for proper citation in the response
        doc_string = f"<doc id='{i}'>\nTitle: {doc.metadata.get('title', 'No Title')}\nURL: {doc.metadata.get('url', 'No URL')}\nContent: {doc.page_content}\n</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def serialize_history(request: ChatRequest):
    """
    Converts the chat history from dict format to LangChain message objects.

    This is necessary because LangChain uses specific message objects
    (HumanMessage, AIMessage) for its chat models.

    Args:
        request: The chat request containing history

    Returns:
        List: Converted chat history as LangChain message objects
    """
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        # Convert user messages
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        # Convert AI messages
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


def create_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    """
    Creates the main RAG (Retrieval-Augmented Generation) chain.

    This chain handles the entire process:
    1. Processes chat history
    2. Retrieves relevant documents
    3. Formats documents
    4. Generates a response using the LLM

    Args:
        llm: The language model for generating responses
        retriever: The retriever for finding relevant documents

    Returns:
        Runnable: The complete RAG chain
    """
    # Chain that handles retrieval logic (direct questions vs. follow-ups)
    retriever_chain = create_retriever_chain(
        llm,
        retriever,
    ).with_config(run_name="FindDocs")

    # Chain that takes retrieved documents and formats them
    context = (
        RunnablePassthrough.assign(docs=retriever_chain)  # Store docs for later use
        .assign(context=lambda x: format_docs(x["docs"]))  # Format docs as context
        .with_config(run_name="RetrieveDocs")
    )

    # Create the chat prompt that includes system instructions, chat history, and user question
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),  # System instructions
            MessagesPlaceholder(variable_name="chat_history"),  # Chat history
            ("human", "{question}"),  # Current question
        ]
    )

    # Chain that takes context and generates a response
    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
        run_name="GenerateResponse"
    )

    # Combine all chains into the final RAG chain
    return (
        RunnablePassthrough.assign(
            chat_history=serialize_history
        )  # Convert chat history
        | context  # Retrieve and format documents
        | response_synthesizer  # Generate the final response
    )


# Initialize only GPT-4o-mini for all operations
# We use temperature=0 for more deterministic responses
# Streaming=True allows for incremental response generation
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

# Initialize retriever and answer chain
# These are the main components that will be used by the API
retriever = get_retriever()
answer_chain = create_chain(llm, retriever)
