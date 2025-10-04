import os
import json
from datetime import datetime
from typing import List, Optional
import traceback
import shutil
from typing import List, Optional, Dict, Any

# --- Nuevas importaciones ---
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# --- Importaciones de LangChain Modificadas ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Cargar variables de entorno ---
load_dotenv()

# --- Configuración Inicial ---
PDF_PATH = "PromptSystem.pdf"
VECTOR_STORE_PATH = "faiss_index"
CHAT_HISTORY_DIR = "chat_histories"

# Crear directorio para historiales si no existe
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# --- Modelos Pydantic ---
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
    description="Una API para interactuar con un asistente legal basado en el Código del Trabajo de Honduras con soporte para historial de chats.",
    version="2.0.0"
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

# --- Funciones de gestión de historial ---
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
    try:
        chat_file = get_chat_file_path(user_id, chat_id)
        chat_data = {
            "chat_id": chat_id,
            "user_id": user_id,
            "messages": messages,
            "updated_at": datetime.now().isoformat(),
            "title": title or "Nueva conversación"
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
        update_user_chats_index(user_id, chat_id, chat_data["title"], chat_data["updated_at"])
        print(f"Historial guardado para chat {chat_id}")
    except Exception as e:
        print(f"Error guardando historial del chat {chat_id}: {e}")

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

# --- Función para limpiar índice FAISS corrupto ---
def clean_faiss_index():
    try:
        if os.path.exists(VECTOR_STORE_PATH):
            print(f"Eliminando índice FAISS en {VECTOR_STORE_PATH}...")
            shutil.rmtree(VECTOR_STORE_PATH)
            print("Índice FAISS eliminado correctamente.")
            return True
    except Exception as e:
        print(f"Error eliminando índice FAISS: {e}")
        return False
    return False

# --- Función para verificar contenido del PDF ---
def verify_pdf_content():
    try:
        print(f"Verificando contenido del PDF: {PDF_PATH}")
        loader = PyMuPDFLoader(PDF_PATH)
        documents = loader.load()
        
        print(f"📄 Total de páginas cargadas: {len(documents)}")
        
        if documents:
            # Mostrar muestra del contenido
            sample_content = documents[0].page_content[:500]
            print(f"📝 Muestra del contenido (primeros 500 chars):")
            print(f"'{sample_content}...'")
            
            # Verificar si contiene contenido sobre vacaciones
            vacation_keywords = ['vacacion', 'vacation', 'descanso', 'días', 'servicio', 'trabajo']
            content_lower = '\n'.join([doc.page_content.lower() for doc in documents[:10]])  # Primeras 10 páginas
            
            found_keywords = [kw for kw in vacation_keywords if kw in content_lower]
            print(f"🔍 Palabras clave encontradas: {found_keywords}")
            
            return True
        else:
            print("⚠️ No se pudo cargar contenido del PDF")
            return False
            
    except Exception as e:
        print(f"❌ Error verificando PDF: {e}")
        return False

# --- Lógica RAG mejorada con diagnósticos ---
def setup_rag_pipeline():
    try:
        # Validate GOOGLE_API_KEY
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY no está configurada en el archivo .env")
        
        print("🔧 Inicializando embeddings de Google...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        # Validate PDF existence
        if not os.path.exists(PDF_PATH):
            raise FileNotFoundError(f"El archivo PDF no se encontró en la ruta: {PDF_PATH}")
        
        # Verify PDF content before processing
        if not verify_pdf_content():
            raise ValueError("El PDF no contiene contenido válido")
        
        print(f"📚 Cargando o creando base de datos vectorial desde '{VECTOR_STORE_PATH}'...")
        
        # Try to load existing vector store
        vector_store = None
        if os.path.exists(VECTOR_STORE_PATH):
            print("🔄 Intentando cargar base de datos vectorial existente...")
            try:
                vector_store = FAISS.load_local(
                    VECTOR_STORE_PATH,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                print("✅ Base de datos vectorial cargada correctamente.")
            except Exception as load_error:
                print(f"❌ Error cargando índice existente: {load_error}")
                print("🧹 El índice puede estar corrupto o ser incompatible. Recreando...")
                clean_faiss_index()
                vector_store = None
        
        # Create new vector store if needed
        if vector_store is None:
            print(f"🆕 Creando nueva base de datos vectorial desde '{PDF_PATH}'...")
            loader = PyMuPDFLoader(PDF_PATH)
            documents = loader.load()
            print(f"📄 Documentos cargados: {len(documents)} páginas")
            
            if not documents:
                raise ValueError("No se pudieron cargar documentos del PDF")
            
            # --- MEJORA 1: Fragmentos más pequeños y precisos ---
            print("📊 Optimizando la división de texto...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", "ARTICULO", " "], # Agregamos "ARTICULO" para mejor división
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)
            print(f"📊 Fragmentos optimizados creados: {len(chunks)}")
            
            if not chunks:
                raise ValueError("No se pudieron crear fragmentos de texto")
            
            print("🔄 Creando embeddings y base vectorial...")
            vector_store = FAISS.from_documents(chunks, embeddings)
            vector_store.save_local(VECTOR_STORE_PATH)
            print(f"💾 Base de datos vectorial guardada en '{VECTOR_STORE_PATH}'.")
        
        # --- MEJORA 2: Estrategia de búsqueda más inteligente (MMR) ---
        print("🧠 Configurando retriever con estrategia MMR...")
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                'k': 5,
                'fetch_k': 20 
            }
        )
        
        print("✅ Retriever configurado correctamente.")
        return retriever
        
    except Exception as e:
        print(f"❌ Error en setup_rag_pipeline: {str(e)}")
        traceback.print_exc()
        raise

def create_query_expansion_chain(llm):
    """
    Crea una cadena para transformar la pregunta del usuario en una consulta de búsqueda más efectiva.
    """
    print("✨ Creando cadena de expansión de consultas...")
    
    # Este prompt le pide al LLM que reformule la pregunta
    query_expansion_prompt = PromptTemplate(
        template="""
        Tu tarea es ayudar a un sistema de búsqueda a encontrar información relevante en el Código del Trabajo de Honduras.
        Dada la siguiente pregunta de un usuario, genera una versión más detallada y técnica de la misma, 
        como si estuviera escrita en el propio código legal. 
        Enfócate en los términos clave, artículos y conceptos legales.

        Pregunta original: "{question}"

        Consulta expandida para búsqueda:
        """,
        input_variables=["question"]
    )
    
    return query_expansion_prompt | llm | StrOutputParser()    

def create_qa_chain(retriever_instance, llm):
    try:
        print("🤖 Creando cadena QA principal con expansión de consulta...")

        # --- PASO 1: Se crea la cadena de expansión ---
        query_expander = create_query_expansion_chain(llm)

        # --- PASO 2: Se mejora el prompt final ---
        # Este prompt es más robusto. Le indica cómo sintetizar información de múltiples fuentes.
        prompt_template = """
        Eres un asistente legal experto en el derecho laboral de Honduras. 
        Tu única fuente de conocimiento es el conjunto de fragmentos del Código del Trabajo de Honduras proporcionado en el CONTEXTO.

        INSTRUCCIONES CRÍTICAS:
        1.  **Analiza CUIDADOSAMENTE todos los fragmentos del contexto.** La respuesta puede requerir combinar información de varios de ellos.
        2.  **Sintetiza una respuesta coherente y completa** basada EXCLUSIVAMENTE en la información proporcionada.
        3.  **Cita el número del artículo específico** (ej. "según el Artículo 325...") si el número del artículo está presente en el contexto.
        4.  Si el contexto no contiene información suficiente para responder, declara explícitamente: "No encontré información específica sobre esta consulta en los fragmentos proporcionados del Código del Trabajo."
        5.  **NO inventes información, no asumas nada y no uses conocimiento externo.**

        CONTEXTO DEL CÓDIGO DEL TRABAJO:
        {context}

        PREGUNTA ORIGINAL DEL USUARIO: 
        {question}

        RESPUESTA PRECISA Y BASADA ÚNICAMENTE EN EL CONTEXTO:
        """
        QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        def format_docs(docs):
            if not docs:
                return "No se encontraron documentos relevantes."
            formatted = [f"Fragmento del Artículo/Sección:\n{doc.page_content}" for doc in docs]
            return "\n\n---\n\n".join(formatted)
        
        # --- PASO 3: Se integra la expansión en la cadena principal ---
        # La pregunta del usuario ahora pasa primero por el expansor
        # La pregunta original se mantiene para el prompt final
        rag_chain_lcel = (
            {
                "context": RunnablePassthrough() | query_expander | retriever_instance | format_docs, 
                "question": RunnablePassthrough()
            }
            | QA_PROMPT
            | llm
            | StrOutputParser()
        )
        
        print("✅ Cadena QA con expansión creada correctamente.")
        return rag_chain_lcel
    except Exception as e:
        print(f"❌ Error en create_qa_chain: {str(e)}")
        traceback.print_exc()
        raise

# --- Evento de Arranque ---
@app.on_event("startup")
def startup_event():
    global rag_chain, retriever
    print("🚀 Iniciando el pipeline RAG...")
    try:
        retriever = setup_rag_pipeline()
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1) # Define el LLM aquí
        rag_chain = create_qa_chain(retriever, llm)
        print("🎉 ¡Pipeline RAG listo y operativo!")
        
        # Test mejorado del pipeline
        test_questions = [
            "¿Cuántos días de vacaciones corresponden?",
            "¿Qué dice sobre las vacaciones?",
            "¿Cuál es el período de vacaciones?"
        ]
        
        for test_q in test_questions:
            try:
                print(f"🧪 Probando: {test_q}")
                answer = rag_chain.invoke(test_q)
                print(f"📝 Respuesta: {answer[:150]}...")
                break  # Solo probar una pregunta exitosa
            except Exception as e:
                print(f"❌ Error en prueba '{test_q}': {str(e)}")
                continue
                
    except Exception as e:
        print(f"❌ Error durante la inicialización del pipeline RAG: {str(e)}")
        traceback.print_exc()
        rag_chain = None
        retriever = None

# --- Endpoints de la API ---
@app.get("/status")
def get_status():
    if rag_chain and retriever:
        return {"status": "ready", "message": "Sistema RAG operativo"}
    return {"status": "loading_error", "message": "Error en la inicialización"}

@app.post("/chat")
def chat_with_rag(request: ChatRequest):
    if not rag_chain or not retriever:
        raise HTTPException(status_code=503, detail="El servicio no está disponible. El pipeline RAG no se ha inicializado.")
    
    print(f"❓ Pregunta recibida: {request.message!r}")
    print(f"🆔 Chat ID: {request.chat_id}")
    print(f"👤 Usuario: {request.user_id}")
    
    try:
        # Sanitize input
        cleaned_message = request.message.strip()
        if not cleaned_message:
            raise ValueError("El mensaje está vacío después de limpiar.")
        
        # Test retriever first
        print("🔍 Probando retriever...")
        try:
            retrieved_docs = retriever.invoke(cleaned_message)
            print(f"📊 Documentos recuperados: {len(retrieved_docs)}")
            
            if retrieved_docs:
                print("📄 Muestra de documentos recuperados:")
                for i, doc in enumerate(retrieved_docs[:2]):
                    print(f"  Doc {i+1}: '{doc.page_content[:100]}...'")
            else:
                print("⚠️ No se recuperaron documentos relevantes")
        except Exception as retriever_error:
            print(f"❌ Error en retriever: {retriever_error}")
            raise HTTPException(status_code=500, detail="Error en la búsqueda de documentos")
        
        # Invoke the chain with error handling
        print("🤖 Invocando cadena RAG...")
        try:
            answer = rag_chain.invoke(cleaned_message)
        except Exception as chain_error:
            print(f"❌ Error en la cadena RAG: {str(chain_error)}")
            if "AssertionError" in str(chain_error) or "dimension" in str(chain_error).lower():
                print("🧹 Error de dimensiones detectado. Limpiando índice FAISS...")
                clean_faiss_index()
                raise HTTPException(
                    status_code=500, 
                    detail="Error de compatibilidad detectado. Por favor, reinicia el servidor para regenerar el índice."
                )
            raise chain_error
        
        # Retrieve source documents for response
        sources_text = ""
        try:
            source_documents = retriever.invoke(cleaned_message)
            if source_documents:
                sources_text = "\n\n---\n\n".join([doc.page_content[:300] + "..." for doc in source_documents[:3]])
        except Exception as retriever_error:
            print(f"❌ Error recuperando fuentes: {str(retriever_error)}")
            sources_text = "Error recuperando fuentes de información."
        
        print(f"✅ Respuesta generada: {answer}")
        return {"response": answer, "sources": sources_text}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error durante el procesamiento: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ocurrió un error al procesar tu pregunta: {str(e)}")

# --- Nuevo endpoint para diagnósticos ---
@app.get("/diagnostics")
@app.get("/diagnostics")
def get_diagnostics():
    try:
        # Add the type hint Dict[str, Any] here
        diagnostics: Dict[str, Any] = {
            "pdf_exists": os.path.exists(PDF_PATH),
            "faiss_index_exists": os.path.exists(VECTOR_STORE_PATH),
            "rag_chain_ready": rag_chain is not None,
            "retriever_ready": retriever is not None,
            "google_api_key_configured": bool(os.getenv("GOOGLE_API_KEY"))
        }

        if diagnostics["pdf_exists"]:
            try:
                loader = PyMuPDFLoader(PDF_PATH)
                docs = loader.load()
                # Now these lines will be valid
                diagnostics["pdf_pages"] = len(docs)
                diagnostics["pdf_content_sample"] = docs[0].page_content[:200] if docs else "Sin contenido"
            except Exception as e:
                diagnostics["pdf_error"] = str(e)

        return diagnostics
    except Exception as e:
        return {"error": str(e)}

@app.get("/chats/{user_id}")
def get_user_chats(user_id: str):
    try:
        chats = get_user_chats_list(user_id)
        return {"chats": chats}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al obtener los chats")

@app.get("/chat/{user_id}/{chat_id}")
def get_chat_history_endpoint(user_id: str, chat_id: str):
    try:
        messages = load_chat_history(user_id, chat_id)
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al obtener el historial del chat")

@app.post("/chat/save")
def save_chat_endpoint(request: SaveChatRequest):
    try:
        save_chat_history(request.user_id, request.chat_id, request.messages, request.title)
        return {"success": True, "message": "Chat guardado correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al guardar el chat")

@app.delete("/chat/{user_id}/{chat_id}")
def delete_chat_endpoint(user_id: str, chat_id: str):
    try:
        success = delete_chat_history(user_id, chat_id)
        if success:
            return {"success": True, "message": "Chat eliminado correctamente"}
        else:
            raise HTTPException(status_code=500, detail="Error al eliminar el chat")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al eliminar el chat")

@app.post("/rebuild-index")
def rebuild_index():
    global rag_chain, retriever
    try:
        print("🔄 Iniciando reconstrucción del índice...")
        clean_faiss_index()
        retriever = setup_rag_pipeline()
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1) # Define el LLM aquí
        rag_chain = create_qa_chain(retriever, llm)
        return {"success": True, "message": "Índice reconstruido correctamente"}
    except Exception as e:
        print(f"❌ Error reconstruyendo índice: {str(e)}")
        raise HTTPException(status_code=500, detail="Error al reconstruir el índice")

@app.get("/")
def read_root():
    return {
        "message": "Asistente Legal RAG API con Historial",
        "version": "2.1.0",
        "status": "ready" if rag_chain and retriever else "loading_error"
    }

# --- Ejecución del servidor ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)