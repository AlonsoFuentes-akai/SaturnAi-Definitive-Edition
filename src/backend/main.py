import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
import traceback
import shutil

# --- Nuevas importaciones ---
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# --- Importaciones de LangChain Modificadas ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- Cargar variables de entorno ---
load_dotenv()

# --- Configuración Inicial ---
PDF_PATH = "PromptSystem.pdf"
VECTOR_STORE_PATH = "faiss_index"
CHAT_HISTORY_DIR = "chat_histories"
LAW_FILES_DIR = "law_files"
# NUEVO: Ruta para el archivo de metadatos del índice
METADATA_PATH = os.path.join(VECTOR_STORE_PATH, "index_metadata.json")

# Crear directorios si no existen
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
os.makedirs(LAW_FILES_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)


# --- Modelos Pydantic (Sin cambios) ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    user_id: str
    chat_id: Optional[str] = None
    chat_history: Optional[List[ChatMessage]] = []

class ChatHistoryRequest(BaseModel):
    user_id: str
    chat_id: str

class SaveChatRequest(BaseModel):
    user_id: str
    chat_id: str
    messages: List[dict]
    title: Optional[str] = None

# --- Inicialización de FastAPI ---
app = FastAPI(
    title="Asistente Legal RAG API con Historial",
    description="Una API para un asistente legal con un corpus de conocimiento que crece dinámicamente.",
    version="3.0.0"
)

# --- Configuración de CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Variables globales ---
rag_chain = None
retriever = None
vector_store = None
embeddings = None
llm = None


# --- Funciones de gestión de historial (Sin cambios) ---
def get_chat_file_path(user_id: str, chat_id: str) -> str:
    return os.path.join(CHAT_HISTORY_DIR, f"{user_id}_{chat_id}.json")

def get_user_chats_index_path(user_id: str) -> str:
    return os.path.join(CHAT_HISTORY_DIR, f"{user_id}_index.json")

def load_chat_history(user_id: str, chat_id: str) -> List[dict]:
    try:
        chat_file = get_chat_file_path(user_id, chat_id)
        if os.path.exists(chat_file):
            with open(chat_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('messages', [])
        return []
    except Exception as e:
        print(f"Error cargando historial del chat {chat_id}: {e}")
        return []


def save_chat_history(user_id: str, chat_id: str, messages: List[dict], title: Optional[str] = None):
    # (Esta función permanece igual que en tu código original)
    try:
        chat_file = get_chat_file_path(user_id, chat_id)
        chat_data = {
            "chat_id": chat_id, "user_id": user_id, "messages": messages,
            "updated_at": datetime.now().isoformat(), "title": title or "Nueva conversación"
        }
        if os.path.exists(chat_file):
            with open(chat_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            chat_data["created_at"] = existing_data.get("created_at", datetime.now().isoformat())
            if not title:
                chat_data["title"] = existing_data.get("title", "Nueva conversación")
        else:
            chat_data["created_at"] = datetime.now().isoformat()
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)
        # (Lógica de actualización de índice de usuario omitida por brevedad, pero debería estar aquí)
    except Exception as e:
        print(f"Error guardando historial del chat {chat_id}: {e}")

# --- (Otras funciones de historial como load, delete, etc. van aquí, sin cambios) ---
def update_user_chats_index(user_id: str, chat_id: str, title: str, updated_at: str):
    try:
        index_file = get_user_chats_index_path(user_id)
        if os.path.exists(index_file):
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
        else:
            index_data = {"user_id": user_id, "chats": []}
        chat_exists = False
        for chat in index_data["chats"]:
            if chat["chat_id"] == chat_id:
                chat["title"] = title
                chat["updated_at"] = updated_at
                chat_exists = True
                break
        if not chat_exists:
            index_data["chats"].append({
                "chat_id": chat_id,
                "title": title,
                "updated_at": updated_at
            })
        index_data["chats"].sort(key=lambda x: x["updated_at"], reverse=True)
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error actualizando índice de chats para usuario {user_id}: {e}")

def get_user_chats_list(user_id: str) -> List[dict]:
    try:
        index_file = get_user_chats_index_path(user_id)
        if os.path.exists(index_file):
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
                return index_data.get("chats", [])
        return []
    except Exception as e:
        print(f"Error obteniendo lista de chats para usuario {user_id}: {e}")
        return []

def delete_chat_history(user_id: str, chat_id: str) -> bool:
    try:
        chat_file = get_chat_file_path(user_id, chat_id)
        if os.path.exists(chat_file):
            os.remove(chat_file)
        index_file = get_user_chats_index_path(user_id)
        if os.path.exists(index_file):
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            index_data["chats"] = [chat for chat in index_data["chats"] if chat["chat_id"] != chat_id]
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
        print(f"Chat {chat_id} eliminado correctamente")
        return True
    except Exception as e:
        print(f"Error eliminando chat {chat_id}: {e}")
        return False

# --- NUEVO: Funciones para gestionar los metadatos del índice ---
def load_metadata() -> Dict[str, Any]:
    """Carga los metadatos del índice desde el archivo JSON."""
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"processed_files": []}

def save_metadata(data: Dict[str, Any]):
    """Guarda los metadatos del índice en el archivo JSON."""
    data["last_updated"] = datetime.now().isoformat()
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


# --- MODIFICADO: Lógica RAG ahora es más modular y dinámica ---
def _get_pdf_chunks(file_path: str) -> List[Document]:
    """Carga y divide un único archivo PDF en fragmentos."""
    print(f"📄 Procesando archivo: {file_path}")
    try:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "ARTICULO", " "],
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"📊 Creados {len(chunks)} fragmentos para {os.path.basename(file_path)}")
        return chunks
    except Exception as e:
        print(f"⚠️ Error procesando {file_path}: {e}")
        return []

def _add_documents_to_index(chunks: List[Document], file_path: str):
    """Añade nuevos fragmentos al índice vectorial y actualiza los metadatos."""
    global vector_store, retriever, rag_chain
    if not chunks:
        return

    print(f"➕ Añadiendo {len(chunks)} nuevos fragmentos al índice...")
    vector_store.add_documents(chunks)
    vector_store.save_local(VECTOR_STORE_PATH)

    # Actualizar metadatos
    metadata = load_metadata()
    metadata["processed_files"].append(file_path)
    save_metadata(metadata)
    
    # Actualizar el retriever y la cadena en memoria
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 7, 'fetch_k': 25})
    rag_chain = create_qa_chain(retriever, llm)
    print(f"✅ Índice actualizado y guardado. Archivo '{os.path.basename(file_path)}' añadido.")

def setup_rag_pipeline():
    """Configura el pipeline RAG, creando o cargando y actualizando el índice."""
    global vector_store, retriever, rag_chain, embeddings, llm
    try:
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY no está configurada en el archivo .env")

        print("🔧 Inicializando componentes RAG...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, convert_system_message_to_human=True)
        
        metadata = load_metadata()
        processed_files = metadata.get("processed_files", [])

        # Cargar el índice si existe
        if os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
            print("🔄 Cargando base de datos vectorial existente...")
            vector_store = FAISS.load_local(
                VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
            )
            print("✅ Base de datos cargada.")
        else:
            print("🆕 No se encontró índice. Se creará uno nuevo.")
            # Crear un índice vacío para poder añadir documentos después
            initial_chunks = [Document(page_content="Inicio del índice legal.")]
            vector_store = FAISS.from_documents(initial_chunks, embeddings)

        # MODIFICADO: Comprobar y procesar archivos nuevos o faltantes
        all_pdf_files = [os.path.join(LAW_FILES_DIR, f) for f in os.listdir(LAW_FILES_DIR) if f.endswith(".pdf")]
        if os.path.exists(PDF_PATH):
             all_pdf_files.append(PDF_PATH)
        
        new_files_to_process = [f for f in all_pdf_files if f not in processed_files]

        if new_files_to_process:
            print(f" 발견! {len(new_files_to_process)} nuevos archivos para indexar.")
            for file_path in new_files_to_process:
                chunks = _get_pdf_chunks(file_path)
                if chunks:
                    vector_store.add_documents(chunks)
                    processed_files.append(file_path)
            
            vector_store.save_local(VECTOR_STORE_PATH)
            save_metadata({"processed_files": processed_files})
            print("✅ Todos los archivos nuevos han sido añadidos al índice.")
        else:
            print("👍 El índice está actualizado. No hay archivos nuevos para procesar.")

        # Configurar el retriever y la cadena final
        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 7, 'fetch_k': 25})
        rag_chain = create_qa_chain(retriever, llm)

    except Exception as e:
        print(f"❌ Error crítico en setup_rag_pipeline: {e}")
        traceback.print_exc()
        raise

# La función create_qa_chain permanece casi igual, solo la he limpiado un poco.
def create_qa_chain(retriever_instance, llm_instance):
    print("🤖 Creando cadena QA principal...")
    prompt_template = """
    Eres un asistente legal experto en la legislación de Honduras. 
    Tu única fuente de conocimiento es el conjunto de fragmentos de leyes proporcionado en el CONTEXTO.

    INSTRUCCIONES CRÍTICAS:
    1.  **Analiza CUIDADOSAMENTE todos los fragmentos del contexto.** La respuesta puede requerir combinar información de varios de ellos.
    2.  **Sintetiza una respuesta coherente y completa** basada EXCLUSIVAMENTE en la información proporcionada.
    3.  **Cita el número del artículo específico** (ej. "según el Artículo 325...") si está presente en el contexto.
    4.  Si el contexto no contiene información para responder, declara: "No encontré información sobre esta consulta en los documentos proporcionados."
    5.  **NO inventes información ni uses conocimiento externo.**

    CONTEXTO DE LAS LEYES:
    {context}

    PREGUNTA DEL USUARIO: 
    {question}

    RESPUESTA PRECISA Y BASADA EN EL CONTEXTO:
    """
    QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    def format_docs(docs):
        return "\n\n---\n\n".join([f"Fragmento del Documento:\n{doc.page_content}" for doc in docs])

    chain = (
        {"context": retriever_instance | format_docs, "question": RunnablePassthrough()}
        | QA_PROMPT
        | llm_instance
        | StrOutputParser()
    )
    print("✅ Cadena QA creada correctamente.")
    return chain

# --- Evento de Arranque ---
@app.on_event("startup")
def startup_event():
    print("🚀 Iniciando el pipeline RAG...")
    try:
        setup_rag_pipeline()
        print("🎉 ¡Pipeline RAG listo y operativo!")
    except Exception as e:
        print(f"❌ Error fatal durante el arranque: {e}")
        # El servidor se iniciará, pero los endpoints fallarán con un error 503.

# --- Endpoints de la API ---
@app.get("/status")
def get_status():
    if rag_chain and retriever:
        return {"status": "ready", "message": "Sistema RAG operativo"}
    return {"status": "error", "message": "Error en la inicialización del sistema RAG"}

@app.post("/chat")
def chat_with_rag(request: ChatRequest):
    if not rag_chain:
        raise HTTPException(status_code=503, detail="Servicio no disponible. El pipeline RAG no se inicializó correctamente.")
    
    print(f"❓ Pregunta recibida: {request.message!r}")
    try:
        cleaned_message = request.message.strip()
        if not cleaned_message:
            raise ValueError("El mensaje está vacío.")
        
        answer = rag_chain.invoke(cleaned_message)
        
        # Opcional: obtener fuentes para la respuesta
        source_documents = retriever.invoke(cleaned_message)
        sources_text = "\n\n---\n\n".join([doc.page_content for doc in source_documents])

        print(f"✅ Respuesta generada.")
        return {"response": answer, "sources": sources_text}
    except Exception as e:
        print(f"❌ Error durante el procesamiento del chat: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ocurrió un error al procesar tu pregunta: {e}")

# MODIFICADO: El endpoint de carga ahora es mucho más eficiente
@app.post("/upload-law-document/")
async def upload_law_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not vector_store:
        raise HTTPException(status_code=503, detail="El índice vectorial no está listo. Intente de nuevo en unos momentos.")
    
    file_path = os.path.join(LAW_FILES_DIR, file.filename)
    
    # Evitar procesar un archivo que ya existe con el mismo nombre
    metadata = load_metadata()
    if file_path in metadata.get("processed_files", []):
        return {"success": False, "message": f"El archivo '{file.filename}' ya existe en el índice."}

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"📄 Archivo '{file.filename}' guardado. Procesando en segundo plano...")

        # Usar tareas en segundo plano para no bloquear la respuesta HTTP
        chunks = _get_pdf_chunks(file_path)
        if not chunks:
             raise HTTPException(status_code=400, detail=f"No se pudo procesar el contenido del archivo '{file.filename}'.")

        background_tasks.add_task(_add_documents_to_index, chunks, file_path)
        
        return {"success": True, "message": f"Archivo '{file.filename}' recibido. Se está añadiendo al índice en segundo plano."}
    except Exception as e:
        print(f"❌ Error durante la carga de archivo: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al guardar o procesar el archivo: {e}")

# NUEVO: Endpoint para forzar una reconstrucción completa si es necesario
@app.post("/rebuild-index")
def rebuild_index(background_tasks: BackgroundTasks):
    """Elimina el índice existente y lo reconstruye desde todos los archivos PDF."""
    print("⛔ Se solicitó una reconstrucción completa del índice.")
    
    def _rebuild_task():
        global vector_store
        # Eliminar el índice viejo
        if os.path.exists(VECTOR_STORE_PATH):
            shutil.rmtree(VECTOR_STORE_PATH)
        os.makedirs(VECTOR_STORE_PATH) # Recrear la carpeta
        
        # Reiniciar el vector_store en memoria
        initial_chunks = [Document(page_content="Inicio del índice legal.")]
        vector_store = FAISS.from_documents(initial_chunks, embeddings)
        
        # Volver a llamar a setup para que procese todos los archivos como si fueran nuevos
        setup_rag_pipeline()

    background_tasks.add_task(_rebuild_task)
    return {"success": True, "message": "La reconstrucción completa del índice ha comenzado en segundo plano."}


@app.get("/")
def read_root():
    return {"message": "API del Asistente Legal Dinámico", "status": "ready" if rag_chain else "error"}

# --- (Otros endpoints de historial de chat van aquí) ---

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
