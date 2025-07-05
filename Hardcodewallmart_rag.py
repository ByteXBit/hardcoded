
# FastAPI RAG endpoint for Walmart products
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- CONFIG ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyA2XmTAncsUZogi6RFkS_oUCbgNIKe5Aaw")
DATA_PATH = "walmart_products.csv"  # Place the CSV in the same directory as this script

# --- FASTAPI APP ---
app = FastAPI()

class AskRequest(BaseModel):
    question: str

# --- GLOBALS (initialized at startup) ---
model = None
collection = None
rag_chain = None

@app.on_event("startup")
def load_resources():
    global model, collection, rag_chain
    # Load data
    if not os.path.exists(DATA_PATH):
        raise RuntimeError(f"CSV file '{DATA_PATH}' not found. Please add it to the script directory.")
    df = pd.read_csv(DATA_PATH)
    documents = [
        f"Product: {row.name}, Brand: {row.brand}, Category: {row.category}, "
        f"Price: ₹{row.price}, Discount: {row.discount}%, Description: {row.description}, "
        f"Stock: {row.stock_quantity} units, Store: {row.store_location}"
        for _, row in df.iterrows()
    ]
    metadatas = [
        {"brand": row.brand, "category": row.category, "store": row.store_location}
        for _, row in df.iterrows()
    ]
    # Embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(documents).tolist()
    # ChromaDB
    client = chromadb.Client()
    collection_ = client.get_or_create_collection(name="walmart_products")
    # Only add if collection is empty
    if collection_.count() == 0:
        collection_.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=[f"prod_{i}" for i in range(len(documents))]
        )
    collection = collection_
    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )
    template = """
You are a helpful assistant for Walmart store data.
Use the following context to answer the user's question.
Answer in 1-2 lines by greeting the customer with meow.

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    rag_chain = LLMChain(llm=llm, prompt=prompt)

@app.post("/ask")
def ask_endpoint(req: AskRequest):
    global model, collection, rag_chain
    if model is None or collection is None or rag_chain is None:
        raise HTTPException(status_code=503, detail="Resources not loaded.")
    query = req.question
    # For demo, try to extract city from question for filtering (optional, can be improved)
    # Here, we just use no filter, or you can parse city from question
    query_embedding = model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    if not results["documents"] or not results["documents"][0]:
        return {"answer": "Meow! Sorry, I couldn't find relevant products."}
    context_chunks = results["documents"][0]
    context = "\n".join(context_chunks)
    response = rag_chain.run({
        "context": context,
        "question": query
    })
    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Render provides PORT env variable
    uvicorn.run("Hardcodewallmart_rag:app", host="0.0.0.0", port=port, reload=False)