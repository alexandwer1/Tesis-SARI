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
load_dotenv() 



def get_pdf_text(pdf_docs):
    """Extrae el texto de los PDFs subidos"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Divide el texto en fragmentos (Chunking Semántico)"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_faiss = os.path.join(ruta_actual, "faiss_index")
    
    if not os.path.exists(ruta_faiss):
        os.makedirs(ruta_faiss)
    
    vector_store = None
    batch_size = 1  
    i = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    while i < len(text_chunks):
        batch = text_chunks[i:i + batch_size]
        status_text.info(f"Procesando fragmento {i} de {len(text_chunks)}...")
        
       
        print(f">>> Trabajando en el fragmento {i} de {len(text_chunks)}...")
        
        try:
            if vector_store is None:
                vector_store = FAISS.from_texts(batch, embedding=embeddings)
            else:
                vector_store.add_texts(batch)
            
           
            i += batch_size
            progress_bar.progress(min(i / len(text_chunks), 1.0))
            
           
            time.sleep(15) 
            
        except Exception as e:
            if "429" in str(e):
                
                status_text.warning("🚨 Bloqueo persistente. Esperando 120 segundos para reintentar...")
                time.sleep(120) 
            else:
                st.error(f"Error inesperado: {e}")
                break

    if vector_store:
        vector_store.save_local(ruta_faiss)
        status_text.success(f"✅ ¡Victoria! Todo el manual de 121 páginas ha sido procesado y guardado.")
        

def get_conversational_chain():
    """Configura el Cerebro del Sistema (Prompt Engineering)"""
    
    prompt_template = """
    Actúa como un Asesor Técnico Pedagógico experto en Inclusión Educativa de la SEP.
    Tu objetivo es ayudar a un docente a reducir las Barreras para el Aprendizaje y la Participación (BAP).
    
    Usa los siguientes fragmentos de los manuales oficiales para responder.
    Si la respuesta no está en el contexto, di "No encuentro información específica en los manuales subidos", no inventes nada.
    
    Contexto (Manuales SEP):
    {context}
    
    Consulta del Maestro:
    {question}
    
    Instrucciones de respuesta:
    1. Sé empático y motivador con el maestro.
    2. Sugiere una estrategia concreta basada en el Diseño Universal para el Aprendizaje (DUA).
    3. Menciona cómo aprovechar los intereses del alumno mencionados.
    4. Cita (si es posible) el documento o concepto oficial.
    
    Respuesta:
    """
    
    model = ChatGoogleGenerativeAI( model="gemini-3-flash-preview", temperature=0.3, google_api_key=st.secrets["GOOGLE_API_KEY"])
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    # 1. Configuramos los embeddings con la clave de Streamlit
    api_key = st.secrets["GOOGLE_API_KEY"]
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=api_key
    )
    print(">>> Cargando base de datos vectorial...") 
    new_db = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    print(">>> Buscando información en el manual...")
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    print(">>> Generando respuesta con Gemini...")
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(">>> ¡Respuesta generada!")
    st.write("### 💡 Estrategia Inclusiva Sugerida:")
    st.write(response["output_text"])
def main():
    
    st.header("🧠 Sistema de Asistencia Docente - Educación Inclusiva")
    st.markdown("---")

    # Barra lateral para subir documentos
    with st.sidebar:
        st.title("📂 Base de Conocimiento")
        st.info("Sube aquí los manuales de la SEP (PDF) para alimentar al sistema.")
        pdf_docs = st.file_uploader("Cargar Manuales", accept_multiple_files=True)
        if st.button("Procesar Documentos"):
            with st.spinner("Analizando y vectorizando manuales..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("¡Sistema actualizado y listo!")

    # Formulario del Maestro
    col1, col2 = st.columns(2)
    
    with col1:
       grado = st.selectbox("Grado Escolar",["1º Preescolar", "2º Preescolar", "3º Preescolar"])
       edad = st.number_input("Edad del alumno", min_value=3, max_value=7, value=3)
    
    with col2:
        diagnostico = st.text_input("Diagnóstico o Condición (Opcional)", placeholder="Ej. TDAH, Autismo, Dislexia...")
        intereses = st.text_input("Intereses del alumno", placeholder="Ej. Dinosaurios, Minecraft, Dibujar...")

    observacion = st.text_area("Descripción de la situación / Barrera observada:", height=100, placeholder="El alumno se levanta constantemente de su silla y pierde el foco cuando hay mucho ruido...")

    if st.button("🔍 Generar Estrategia Inclusiva"):
        if observacion:
            # Creamos el Prompt completo con los datos del formulario
            consulta_completa = f"""
            Soy docente de {grado}. Tengo un alumno de {edad} años.
            Diagnóstico: {diagnostico}.
            Intereses: {intereses}.
            Situación observada: {observacion}.
            ¿Qué estrategia de intervención sugieres basada en los manuales?
            """
            user_input(consulta_completa)
        else:
            st.warning("Por favor, describe la situación para poder ayudarte.")

if __name__ == "__main__":

    main()



