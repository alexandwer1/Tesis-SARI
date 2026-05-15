import streamlit as st
import time
import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from openpyxl import Workbook
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.question_answering import load_qa_chain

# ==============================
# CONFIGURACIÓN
# ==============================
load_dotenv()

def get_api_key():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        key = st.secrets.get("GOOGLE_API_KEY")
    return key

# ==============================
# BASE DE DATOS HISTORIAL
# ==============================
def init_db():
    conn = sqlite3.connect("historial_casos.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS casos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT,
            nivel TEXT,
            grado TEXT,
            edad INTEGER,
            diagnostico TEXT,
            intereses TEXT,
            descripcion TEXT,
            estrategia TEXT
        )
    """)
    conn.commit()
    conn.close()

def guardar_caso(nivel, grado, edad, diagnostico, intereses, descripcion, estrategia):
    conn = sqlite3.connect("historial_casos.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO casos (fecha, nivel, grado, edad, diagnostico, intereses, descripcion, estrategia)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        nivel,
        grado,
        edad,
        diagnostico,
        intereses,
        descripcion,
        estrategia
    ))
    conn.commit()
    conn.close()

def obtener_casos():
    conn = sqlite3.connect("historial_casos.db")
    c = conn.cursor()
    c.execute("SELECT * FROM casos ORDER BY fecha DESC")
    datos = c.fetchall()
    conn.close()
    return datos

def mostrar_historial():
    datos = obtener_casos()
    st.markdown("## 📚 Historial de Casos")

    if not datos:
        st.info("Aún no hay casos registrados.")
        return

    for caso in datos:
        with st.expander(f"Caso #{caso[0]} - {caso[1]} - {caso[3]}"):
            st.write(f"Nivel: {caso[2]}")
            st.write(f"Edad: {caso[4]}")
            st.write(f"Diagnóstico: {caso[5]}")
            st.write(f"Intereses: {caso[6]}")
            st.write("Descripción:")
            st.write(caso[7])
            st.write("Estrategia Generada:")
            st.write(caso[8])

def exportar_a_excel():
    datos = obtener_casos()

    if not datos:
        st.warning("No hay datos para exportar.")
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "Historial Casos"

    encabezados = [
        "ID", "Fecha", "Nivel", "Grado", "Edad",
        "Diagnóstico", "Intereses", "Descripción", "Estrategia"
    ]
    ws.append(encabezados)

    for fila in datos:
        ws.append(fila)

    archivo = "historial_casos.xlsx"
    wb.save(archivo)

    with open(archivo, "rb") as f:
        st.download_button(
            label="⬇ Descargar Excel",
            data=f,
            file_name="historial_casos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ==============================
# PROCESAMIENTO PDF
# ==============================
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=get_api_key()
    )

    vector_store = None
    batch_size = 80

    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]

        if vector_store is None:
            vector_store = FAISS.from_texts(batch, embedding=embeddings)
        else:
            vector_store.add_texts(batch)

        if (i + batch_size) < len(text_chunks):
            time.sleep(60)

    vector_store.save_local("faiss_index")

# ==============================
# LLM
# ==============================
def get_conversational_chain():
    prompt_template = """
    Actúa como un Asesor Técnico Pedagógico experto en Inclusión Educativa de la SEP.
    Ayuda al docente a reducir las Barreras para el Aprendizaje y la Participación (BAP).

    Contexto:
    {context}

    Pregunta:
    {question}

    Respuesta experta:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.3,
        google_api_key=get_api_key()
    )

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def generar_estrategia(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=get_api_key()
    )

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# ==============================
# INTERFAZ PRINCIPAL
# ==============================
def main():
    init_db()

    st.set_page_config(page_title="Sistema Asistencia Docente", layout="wide", page_icon="🎓")
    st.header("🧠 Sistema de Asistencia Docente - Educación Inclusiva")
    st.markdown("---")

    with st.sidebar:
        st.title("📂 Base de Conocimiento")
        pdf_docs = st.file_uploader("Cargar Manuales", accept_multiple_files=True)

        if st.button("Procesar Documentos"):
            if pdf_docs:
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Manuales procesados.")
            else:
                st.warning("Sube archivos primero.")

        st.markdown("---")

        if st.button("📚 Ver Historial"):
            mostrar_historial()

        if st.button("📊 Exportar a Excel"):
            exportar_a_excel()

    col1, col2 = st.columns(2)

    with col1:
        nivel = st.selectbox("Nivel Educativo", ["Preescolar", "Primaria"])

        if nivel == "Preescolar":
            grado = st.selectbox("Grado", ["1º", "2º", "3º"])
            edad = st.number_input("Edad", 3, 7, 3)
        else:
            grado = st.selectbox("Grado", ["1º", "2º", "3º", "4º", "5º", "6º"])
            edad = st.number_input("Edad", 6, 12, 6)

    with col2:
        diagnostico = st.text_input("Diagnóstico (Opcional)")
        intereses = st.text_input("Intereses del alumno")

    observacion = st.text_area("Descripción de la situación:", height=150)

    if st.button("🔍 Generar Estrategia Inclusiva"):
        if observacion:
            consulta = f"Alumno de {nivel} {grado}, {edad} años. Diagnóstico: {diagnostico}. Intereses: {intereses}. Problema: {observacion}"
            estrategia = generar_estrategia(consulta)

            st.markdown("### 💡 Estrategia Inclusiva Sugerida:")
            st.write(estrategia)

            guardar_caso(nivel, grado, edad, diagnostico, intereses, observacion, estrategia)
            st.success("Caso guardado en historial.")
        else:
            st.warning("Describe la situación.")

if __name__ == "__main__":
    main()