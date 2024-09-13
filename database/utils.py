import io
import pdfplumber
import docx
from langchain_experimental.text_splitter import SemanticChunker
from chromadb.utils import embedding_functions
import subprocess
from pathlib import Path

def process_pdf(content):
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        text += " | ".join(str(cell) for cell in row) + "\n"
    except Exception as e:
        raise ValueError(f"Error processing PDF: {e}")

    return text

def pdf2markdown(input_file: str, output_dir: str) -> Path:
    file_name = Path(input_file).stem
    output_path = Path(output_dir) / file_name
    # Запускаем marker_single для конвертации PDF в Markdown
    subprocess.run(f"marker_single {input_file} {output_dir} --batch_multiplier 2 --langs English,Russian", shell=True, check=True)
    # Форматируем Markdown файл
    subprocess.run(f"mdformat {output_path / (file_name + '.md')}", shell=True, check=True)
    # Возвращаем путь к папке с результатами
    return output_path

def process_docx(content):
    doc = docx.Document(io.BytesIO(content))
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def process_txt(content):
    return content.decode("utf-8")

def process_document(filename, content):
    if filename.endswith(".pdf"):
        return process_pdf(content)
    elif filename.endswith(".docx"):
        return process_docx(content)
    elif filename.endswith(".txt"):
        return process_txt(content)
    else:
        raise ValueError("Unsupported document format")

def add_document_to_db(document_data, collection, text_splitter):
    chunks = text_splitter.split_text(document_data)
    for i, chunk in enumerate(chunks):
        chunk_id = f"{collection.name}-{i+1}"
        collection.add(
            documents=[chunk],
            metadatas=[{"source": "upload", "chunk_id": chunk_id}],
            ids=[chunk_id]
        )

def search_in_db(query_text, collection, num):
    try:
        results = collection.query(query_texts=[query_text], n_results=num)
        return results
    except Exception as e:
        raise e
