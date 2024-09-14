import io
import pdfplumber
import docx
from langchain_experimental.text_splitter import SemanticChunker
from chromadb.utils import embedding_functions
import subprocess
from pathlib import Path
import re
import os
import uuid
cnt = 0
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

def img2txt(cnt_dir, inpt_md, outpt_dir):
    global cnt
    with open(inpt_md) as text_file:
        txt = text_file.read()
    res = {}
    for i in os.listdir(cnt_dir):
        name = i[:i.find('.')]
        if i[i.rfind('.') + 1:] != "png":
            continue
        while re.search(f'!\[{name}\.png\]\({name}\.png\)', txt, re.IGNORECASE):
            x = re.search(f'!\[{name}\.png\]\({name}\.png\)', txt, re.IGNORECASE)
            txt = txt.replace(txt[x.start(): x.end()], f"$$$picture_{cnt}$$$\nЗдесь находится изображение, на которое могут и можно ссылаться.")
            os.replace(f"{cnt_dir}/{i}", f"{outpt_dir}/images/picture_{cnt}.png")
            cnt += 1
    with open(inpt_md, 'w') as out:
        out.write(txt)
    return txt

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
        chunk_id = uuid.uuid4()
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
