import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from google.oauth2 import service_account


# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
google_api_key = os.getenv('GOOGLE_AI_STUDIO_API_KEY')
google_credentials_key = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
credentials = service_account.Credentials.from_service_account_file(google_credentials_key)
# Use the API key in your application
print(google_api_key)
print(google_credentials_key)


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    credentials=credentials
)

messages = [
    ('system', 'You are a helpful assistant that translates English to French. Translate the user sentence.'),
    ('human', 'I love programming.')
]

ai_msg = llm.invoke(messages)

print(ai_msg.content)