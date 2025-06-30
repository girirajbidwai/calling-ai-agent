import os
from flask import Flask, request, Response
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from typing import TypedDict, List
import numpy as np

# Environment variables
os.environ["GOOGLE_API_KEY"] = "AIzaSyB-b31_MOH9WlYfg70W0yKGITDycZ85KJ8"
os.environ["PINECONE_API_KEY"] = "pcsk_3M2zCN_nMqUvwtnMW4LdTa5w9QzjyjNEz5NiCNowqAc1FJnj64Yx5gqKnPCY5MS9XV3BK"
os.environ["TWILIO_ACCOUNT_SID"] = "AC4e827d688a8c99019fe75e54b50df0f1"
os.environ["TWILIO_AUTH_TOKEN"] = "403b58d924a623356765ddaa6021863b"
os.environ["TWILIO_PHONE_NUMBER"] = "12292672049"

# Initialize Twilio client
twilio_client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])

# Initialize Flask app
app = Flask(__name__)

# Initialize LangChain and Pinecone clients
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Pinecone index setup
INDEX_NAME = "rag-index"
if INDEX_NAME not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=INDEX_NAME,
        dimension=384,  # Dimension of all-MiniLM-L6-v2 embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pinecone_client.Index(INDEX_NAME)

# State definition
class State(TypedDict):
    query: str
    documents: List[Document]
    similarity_score: float
    response: str
    is_task: bool
    query_embedding: List[float]

# Document ingestion
def ingest_documents(docs: List[str]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.create_documents(docs)
    for i, chunk in enumerate(chunks):
        embedding = embedder.encode(chunk.page_content).tolist()
        index.upsert(vectors=[(f"doc_{i}", embedding, {"text": chunk.page_content})])

# Node: Query Embedding
def embed_query(state: State) -> State:
    embedding = embedder.encode(state["query"]).tolist()
    state["query_embedding"] = embedding
    return state

# Node: Vector Search
def vector_search(state: State) -> State:
    query_embedding = state.get("query_embedding", [])
    if not query_embedding:
        state["documents"] = []
        state["similarity_score"] = 0.0
        return state
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    state["documents"] = [
        Document(page_content=match["metadata"]["text"])
        for match in results["matches"]
    ]
    state["similarity_score"] = results["matches"][0]["score"] if results["matches"] else 0.0
    return state

# Node: Decision Node
def decide_path(state: State) -> str:
    task_keywords = ["calculate", "generate", "write", "create", "perform"]
    state["is_task"] = any(keyword in state["query"].lower() for keyword in task_keywords)
    if state["is_task"]:
        return "llm_direct"
    elif state["similarity_score"] >= 0.7:
        return "rag_response"
    else:
        return "llm_direct"

# Node: RAG Response
def rag_response(state: State) -> State:
    context = "\n".join([doc.page_content for doc in state["documents"]])
    prompt = f"Context:\n{context}\n\nQuestion: {state['query']}\nAnswer:"
    response = llm.invoke(prompt)
    state["response"] = response.content
    return state

# Node: Direct LLM Response
def llm_direct(state: State) -> State:
    response = llm.invoke(state["query"])
    state["response"] = response.content
    return state

# Build the LangGraph workflow
workflow = StateGraph(State)
workflow.add_node("embed_query", embed_query)
workflow.add_node("vector_search", vector_search)
workflow.add_node("rag_response", rag_response)
workflow.add_node("llm_direct", llm_direct)
workflow.set_entry_point("embed_query")
workflow.add_edge("embed_query", "vector_search")
workflow.add_conditional_edges(
    "vector_search",
    decide_path,
    {
        "rag_response": "rag_response",
        "llm_direct": "llm_direct"
    }
)
workflow.add_edge("rag_response", END)
workflow.add_edge("llm_direct", END)

# Compile the graph
graph = workflow.compile()

# Process query through LangGraph
def process_query(query: str) -> str:
    result = graph.invoke({
        "query": query,
        "documents": [],
        "similarity_score": 0.0,
        "response": "",
        "is_task": False,
        "query_embedding": []
    })
    return result["response"]

# Flask route for incoming voice calls
@app.route("/voice", methods=["POST"])
def voice():
    response = VoiceResponse()
    response.say("Welcome! Please ask your question after the beep, or say hang up to end.")
    # Use <Gather> to collect speech input
    gather = response.gather(
        input="speech",
        action="/process_speech",
        method="POST",
        speech_timeout="auto",
        language="en-US"
    )
    response.redirect("/voice")  # Redirect if no input
    return str(response)

# Flask route to process transcribed speech
@app.route("/process_speech", methods=["POST"])
def process_speech():
    response = VoiceResponse()
    speech_result = request.values.get("SpeechResult", "").strip()
    
    if any(phrase in speech_result.lower() for phrase in ["hang up", "end call", "goodbye", "stop call"]):
        # 1) Say goodbye
        response.say("Okay, goodbye.")

        response.pause(length=20)
        # 2) TwiML attempt
        response.hangup()

        # 3) REST API kill (fallback)
        call_sid = request.values.get("CallSid")
        twilio_client.calls(call_sid).update(status="completed")

        return Response(str(response), mimetype="application/xml")

    if not speech_result:
        response.say("Sorry, I didn't catch that. Please try again.")
        response.redirect("/voice")
        return str(response)
    
    # Check if user said "end" to terminate the call
    
    
    # Process the transcribed query through LangGraph
    answer = process_query(speech_result)
    
    # Limit response length for voice (Twilio <Say> has a 4000-character limit)
    answer = answer[:200]
    
    # Use Twilio's <Say> for TTS
    response.say(answer, voice="Polly.Joanna")  # Use a high-quality voice
    
    # Redirect to /voice to allow another question
    response.redirect("/voice")
    
    return str(response)

# Example documents for ingestion
sample_docs = [
    "The capital of France is Paris.",
    "Python is a versatile programming language used for web development and data science.",
    "The Eiffel Tower is a famous landmark in Paris."
]
ingest_documents(sample_docs)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5000)