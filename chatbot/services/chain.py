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

from chatbot.services.ingest import get_embeddings_model

# System prompt template that instructs the LLM how to respond to user questions
# It defines the response format, tone, and how to handle citations
RESPONSE_TEMPLATE = """\
You are a restaurant business startup expert and consultant, tasked with answering any questions \
about restaurant business startups.

Generate a comprehensive and informative answer of 200 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use the same tone as the \
search results. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

If the user asks a question in Korean, respond in Korean. Match your language to the user's language.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer. If responding in Korean, say "음, 잘 모르겠네요."

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
"""

# Template for rephrasing follow-up questions based on chat history
# Used to convert follow-up questions into standalone questions
REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""

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
