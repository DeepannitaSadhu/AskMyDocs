import json
import os
import io
import chromadb
import fitz  # PyMuPDF
from PIL import Image
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
import google.generativeai as genai

# ==========================================
# URGENT FRAMEWORK BUG FIX (MONKEY PATCH)
# ==========================================
# This forcibly prevents the ADK telemetry logger from crashing 
# when it encounters raw file bytes in the chat history.
_original_json_default = json.JSONEncoder.default

def _safe_json_default(self, obj):
    if isinstance(obj, bytes):
        return "<bytes_omitted_for_telemetry>"
    try:
        return _original_json_default(self, obj)
    except TypeError:
        return str(obj)

json.JSONEncoder.default = _safe_json_default
# ==========================================

# ==========================================
# INITIALIZATIONS & DIRECTORIES
# ==========================================
# 1. Define where the agent should look for PDFs
DOCUMENTS_DIR = "./documents"
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# 2. Define where the agent should save extracted images
IMAGES_DIR = "./extracted_images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# 3. Initialize Databases and Models
client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection = client.get_or_create_collection(name="multimodal_rag")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", "AIzaSyDSd8QF9nHFqmes_OCP8DyFHa7YjIxoLTE"))
gemini = genai.GenerativeModel("gemini-2.5-flash")


# ==========================================
# FUNCTIONS / TOOLS
# ==========================================

def extract_pdf(file_name: str) -> dict:
    """
    Extracts text and images from a PDF file located in the agent's documents folder.
    
    Args:
        file_name (str): The exact name of the PDF file (e.g., 'sample.pdf'). Do NOT use full paths.
        
    Returns:
        dict: A dictionary containing extracted 'texts', 'image_paths' (strings), or an 'error' message.
    """
    clean_file_name = os.path.basename(file_name)
    file_path = os.path.join(DOCUMENTS_DIR, clean_file_name)

    if not os.path.exists(file_path):
        return {
            "error": f"File '{clean_file_name}' not found in the documents folder. Please ask the user to verify the file name."
        }

    doc = fitz.open(file_path)
    texts = []
    image_paths = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            texts.append((page_num, text.strip()))

        for img_index, img in enumerate(page.get_images()):
            xref = img[0]
            base = doc.extract_image(xref)

            image_bytes = base["image"]
            image_ext = base["ext"] 
            
            image_filename = os.path.join(IMAGES_DIR, f"{clean_file_name}_page_{page_num}_img_{img_index}.{image_ext}")
            
            with open(image_filename, "wb") as f:
                f.write(image_bytes)

            image_paths.append((page_num, image_filename))

    return {"texts": texts, "image_paths": image_paths}


def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """
    Splits a large block of text into smaller, manageable chunks.
    
    Args:
        text (str): The input text string to be chunked.
        chunk_size (int, optional): The maximum number of characters per chunk. Defaults to 500.
        
    Returns:
        list[str]: A list containing the chunked text strings.
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
    return chunks


def caption_image(image_path: str) -> str:
    """
    Generates a descriptive text caption for a given image file.
    
    Args:
        image_path (str): The local file path to the image to be captioned.
        
    Returns:
        str: The generated text caption for the image.
    """
    if not os.path.exists(image_path):
        return "Error: Image file not found."

    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption


def store_in_chroma(text: str, metadata: dict, doc_id: str) -> str:
    """
    Generates embeddings for text and stores it in the ChromaDB vector database.
    
    Args:
        text (str): The text content to store.
        metadata (dict): Additional metadata associated with the text.
        doc_id (str): A unique identifier for this document chunk.
        
    Returns:
        str: A confirmation string "stored" upon success.
    """
    embedding = embedding_model.encode(text).tolist()

    collection.add(
        documents=[text],
        metadatas=[metadata],
        ids=[doc_id],
        embeddings=[embedding]
    )

    return "stored"


def ambiguity_checker(query: str) -> str:
    """
    Generates a prompt to determine if a user query is ambiguous or clear.
    
    Args:
        query (str): The user's input query.
        
    Returns:
        str: The formatted prompt to be sent to the LLM for ambiguity checking.
    """
    prompt = f"""
    Determine if the query is ambiguous.

    Query: {query}

    A query is ambiguous if:
    - it is too short
    - it contains unclear references like "this", "that", "it"
    - it does not specify a document or topic
    - it could refer to multiple documents

    If the query is clear, respond with:
    CLEAR
    If the query is not clear, respond with:
    AMBIGUOUS
    """
    return prompt


def Retrieval(query: str) -> dict:
    """
    Retrieves the most relevant document chunks from ChromaDB based on the query.
    
    Args:
        query (str): The search query.
        
    Returns:
        dict: A dictionary containing the retrieval type and relevant context.
    """
    query_embedding = (embedding_model.encode(query)).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not documents:
        return {
            "type": "no_results",
            "message": "No relevant information found."
        }

    # SAFE DICTIONARY LOOKUP: Checks for multiple variations the LLM might have used
    doc_names = []
    for meta in metadatas:
        if meta: # Ensure meta isn't None
            name = meta.get("document_name") or meta.get("document") or meta.get("documents") or "Unknown Document"
            doc_names.append(name)
            
    unique_docs = list(set(doc_names))

    if len(unique_docs) > 1 and "Unknown Document" not in unique_docs:
        return {
            "type": "ambiguity",
            "documents": unique_docs
        }

    context = "\n".join(documents)

    return {
        "type": "context",
        "document": unique_docs[0] if unique_docs else "Unknown Document",
        "context": context
    }


def answer_generator(query: str, context: str) -> str:
    """
    Generates a prompt to instruct the LLM to answer a query using ONLY the provided context.
    
    Args:
        query (str): The user's question.
        context (str): The retrieved textual context from the database.
        
    Returns:
        str: The formatted prompt to send to the LLM.
    """
    prompt = f"""
    Answer ONLY using the context below.
    If the answer is not in the context, say "Not found in document".

    Context:
    {context}

    Question:
    {query}

    Important: DO NOT GENERATE ON YOUR OWN.
    """
    return prompt


def relevance_checker(query: str, answer: str) -> str:
    """
    Generates a prompt to verify if an generated answer is actually relevant to the query.
    
    Args:
        query (str): The original user question.
        answer (str): The generated answer to be evaluated.
        
    Returns:
        str: The formatted prompt to send to the LLM for evaluation.
    """
    prompt = f"""
    Is this answer relevant to the question?

    Question: {query}
    Answer: {answer}

    Answer ONLY: RELEVANT or IRRELEVANT
    """
    return prompt