from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load vector store
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings, collection_name="co2_activities")

# Initialize LLM
llm = OllamaLLM(model="tinyllama")

# Define prompt template
template = """
You are an AI agent specialized in CO2 reduction awareness and action planning. Based on the user's query and the retrieved information about activities and their CO2 emissions, provide actionable recommendations to reduce carbon footprint.

Retrieved information:
{context}

User query: {question}

Instructions:
- Only use the retrieved information to inform your response. If the query is unrelated to CO2 emissions, politely redirect to CO2 reduction topics.
- Identify the specific activity from the query and provide its emission estimate if available in the context.
- Suggest practical, actionable steps to reduce emissions for that activity.
- Provide estimated reductions where possible, based on general knowledge or context.
- Include long-term recommendations.
- Be user-friendly, encouraging, and concise.

Response structure:
- Current estimated emission for the activity.
- Suggested actions with reductions.
- Long-term recommendations.
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Create retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": prompt}
)

def get_recommendations(query):
    return qa_chain.invoke(query)
