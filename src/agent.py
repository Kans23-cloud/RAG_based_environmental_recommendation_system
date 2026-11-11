from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
try:
    # pyrefly: ignore[missing-import]
    from langchain.chains import RetrievalQA # type: ignore
except ImportError:
    from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load vector store
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings, collection_name="co2_activities")

# Initialize LLM
llm = OllamaLLM(
    model="tinyllama",
    temperature=0.1,
    num_predict=350,
    stop=["Remarks:", "Note:", "Remember:", "\n\n4.", "User:", "Question:", "In summary"]
)

# Define prompt template
template = """You are an AI environmental assistant.
Based on the Context, answer the User Query in 2 short, complete recommendations:
- State the user's current emission value (kg/day).
- Recommend cleaner alternatives citing their exact emissions (kg/day).
Finish each sentence completely. Do not repeat raw metadata or category notes.

Retrieved Context:
{context}

User Query: {question}

Recommendations:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# Create retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": prompt}
)

def get_recommendations(query):
    response = qa_chain.invoke(query)
    if isinstance(response, dict):
        result = response.get("result", "")
    else:
        result = str(response)

    result = result.strip()
    for prefix in ["Recommendations:", "Actionable Recommendations:", "AI:", "Assistant:"]:
        if result.startswith(prefix):
            result = result[len(prefix):].strip()

    return result
