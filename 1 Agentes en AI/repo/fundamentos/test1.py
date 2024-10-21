import os
from dotenv import load_dotenv
from langchain_huggingface.llms import HuggingFacePipeline

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
api_key = os.getenv('GOOGLE_AI_STUDIO_API_KEY')

# Use the API key in your application
# Use the API key in your application
if api_key:
    print("api_key=" + api_key)  # or use it in your API calls
else:
    print("API key not found.")

hf = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 10},
)

# Define the question
question = "What is electroencephalography?"

# Invoke the model with the question
response = hf.invoke(question)

# Print the response
print(response)