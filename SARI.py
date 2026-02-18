import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.question_answering import load_qa_chain
import time
import os

st.set_page_config(page_title="Sistema Experto Inclusión SEP", layout="wide", page_icon="🎓")

# Función para cargar PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Chunking
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks

# Crear Vector Store (FAISS)
def get_vector_store(text_chunks):
    # Usamos embedding-001 que es más estable para evitar errores 404
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_faiss = os.path.join(ruta_actual, "faiss_index")
    
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index") # Guardamos en la raíz para facilitar lectura

# Configurar el Cerebro (LLM)
def get_conversational_chain():
    prompt_template = """
    Actúa como un Asesor Técnico Pedagógico experto en Inclusión Educativa de la SEP.
    Tu objetivo es ayudar a un docente a reducir las Barreras para el Aprendizaje y la Participación (BAP).
    
    Usa los siguientes fragmentos de los manuales oficiales para responder.
    Si la respuesta no está en el contexto, di "No encuentro información específica en los manuales subidos".
    
    Contexto:
    {context}
    
    Consulta:
    {question}
    
    Respuesta:
    """
    
    # Usamos gemini-1.5-flash que es el estándar actual estable
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0.3, 
        google_api_key=st.secrets["GOOGLE_API_KEY"]
    )
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Procesar pregunta del usuario
def user_input(user_question):
    # Configuración CRÍTICA para evitar errores de API
    api_key = st.secrets["GOOGLE_API_KEY"]
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", # Debe coincidir con el de creación
        google_api_key=api_key
    )
    
    # Cargar base de datos con manejo de seguridad
    try:
        new_db = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        docs = new_db.similarity_search(user_question)
        
        chain = get_conversational_chain()
        
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        st.markdown("### 💡 Estrategia Inclusiva Sugerida:")
        st.write(response["output_text"])
        
    except Exception as e:
        st.error(f"Error al procesar la consulta: {str(e)}")
        st.info("Intenta subir los manuales nuevamente si el error persiste.")

def main():
    st.header("🧠 Sistema de Asistencia Docente - Educación Inclusiva")
    st.markdown("---")

    with st.sidebar:
        st.title("📂 Base de Conocimiento")
        pdf_docs = st.file_uploader("Cargar Manuales", accept_multiple_files=True)
        if st.button("Procesar Documentos"):
            with st.spinner("Analizando manuales..."):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("¡Manuales procesados correctamente!")
                else:
                    st.warning("Por favor sube al menos un PDF.")

    col1, col2 = st.columns(2)
    with col1:
        grado = st.selectbox("Grado",["1º Preescolar", "2º Preescolar", "3º Preescolar"])
        edad = st.number_input("Edad", 3, 7, 3)
    with col2:
        diagnostico = st.text_input("Diagnóstico", placeholder="Ej. TDAH")
        intereses = st.text_input("Intereses", placeholder="Ej. Dinosaurios")

    observacion = st.text_area("Situación observada:", height=100)

    if st.button("🔍 Generar Estrategia"):
        if observacion:
            consulta = f"Docente de {grado}, alumno de {edad} años. Diagnóstico: {diagnostico}. Intereses: {intereses}. Situación: {observacion}."
            user_input(consulta)
        else:
            st.warning("Describe la situación primero.")

if __name__ == "__main__":
    main()



