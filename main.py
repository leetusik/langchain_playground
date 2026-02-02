from dotenv import load_dotenv

load_dotenv()

from langfuse.langchain import CallbackHandler
 
langfuse_handler = CallbackHandler()

from langchain_google_genai import ChatGoogleGenerativeAI


model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    thinking_level="low",  # For faster, lower-latency responses
)

response = model.invoke("who is the president of the south korea? and how old is he?", callbacks=[langfuse_handler] )
print(response)