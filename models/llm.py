from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import GOOGLE_API_KEY


def get_llm():

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        convert_system_message_to_human=True
    )

    return llm