import os

import streamlit as st
from dotenv import load_dotenv

# from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate

# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()


st.set_page_config(
    page_title="Structured Output. binding tools",
    page_icon="🧪",
)

# genai doesn't support streaming? fuck
# class ChatCallbackHandler(BaseCallbackHandler):
#     message = ""

#     def on_llm_start(self, *args, **kwargs):
#         self.message_box = st.empty()

#     def on_llm_end(self, *args, **kwargs):
#         save_message(self.message, "ai")

#     def on_llm_new_token(self, token, *args, **kwargs):
#         self.message += token
#         self.message_box.markdown(self.message)


google_api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    # streaming=True,
    # callbacks=[ChatCallbackHandler()],
)


class Schema(BaseModel):
    """Always use this tool to structure your response to the user."""

    answer: str = Field(..., description="The answer to the user's question")
    expected_following_question: str = Field(
        ..., description="A followup question the user could ask"
    )


mwso = llm.with_structured_output(Schema)
mwt = llm.bind_tools([Schema])


def save_message(message, role):
    st.session_state.messages.append({"role": role, "message": message})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        # print(message["message"], message["role"])
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful relationship advisor. 
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("TestPage")

st.markdown(
    """
    Ask me anything about relationships!
    """
)

with st.sidebar:
    hello = st.text_input("go?!")

if hello:
    send_message("Hello mister! What can I do for you?", "ai", save=False)

    paint_history()

    message = st.chat_input("Ask me anything about relationships!")
    if message:
        send_message(message, "human")
        chain = prompt | mwt
        # with st.chat_message("ai"):
        result = chain.invoke(message)
        print(result, type(result))
        send_message(result, "ai")
        # send_message(result.expected_following_question, "ai")
else:
    st.session_state["messages"] = []
