from pathlib import Path
import pandas as pd
import torch
import os
import re
import csv
from datetime import datetime, timedelta
from jose import JWTError, jwt
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from deep_translator import GoogleTranslator
from unstructured.partition.doc import partition_doc
from unstructured.partition.docx import partition_docx
from unstructured.partition.pdf import partition_pdf
from fastapi import HTTPException
import unicodedata
from transformers import TextIteratorStreamer
from threading import Thread
import hashlib



# --- Global Variables ---
SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES = "YOUR_SECRET_KEY", "HS256", 60
device = "cuda" if torch.cuda.is_available() else "cpu"
USER_CSV = "users.csv"
# --- Models ---
embed_model_id, llm_model_id = "Qwen/Qwen3-Embedding-0.6B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_id, trust_remote_code=True)
embed_model = AutoModel.from_pretrained(embed_model_id, trust_remote_code=True).to(device).eval()
embed_dim = embed_model.config.hidden_size
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_id, trust_remote_code=True, device_map="auto")

df_chunks_128 = pd.read_csv("chunk_viet/chunk_viet_256_128_translated.csv") if Path("chunk_viet/chunk_viet_256_128_translated.csv").exists() else pd.DataFrame()
embedding_cols = [str(i) for i in range(embed_dim)]

# --- Helper Functions ---
def normalize_filename(name): 
    return unicodedata.normalize("NFC", name)

def save_csv(df, path): 
    df.to_csv(path, index=False)

def next_id(df, col="id"): 
    return int(df[col].max()) + 1 if not df.empty else 1

def hash_password(password: str) -> str:
    """Hash mật khẩu bằng SHA256."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def register_user(username: str, password: str):
    """Thêm user mới vào users.csv nếu chưa tồn tại."""
    df_users = pd.read_csv(USER_CSV, dtype=str) if Path(USER_CSV).exists() else pd.DataFrame(
        columns=["id", "username", "role", "password"]
    )
    
    # Kiểm tra trùng username
    if not df_users[df_users["username"] == username].empty:
        raise HTTPException(400, "Tên đăng nhập đã tồn tại.")
    
    user_id = next_id(df_users, "id")
    new_user = {
        "id": user_id,
        "username": username,
        "role": "user",
        "password": hash_password(password),
    }
    df_users = pd.concat([df_users, pd.DataFrame([new_user])], ignore_index=True)
    save_csv(df_users, USER_CSV)
    return {"id": user_id, "username": username, "role": "user"}

def get_abs_path(file_path: str) -> Path:
    path = Path("data") / normalize_filename(file_path)
    if not path.exists(): raise HTTPException(404, f"File not found: {path}")
    return path

def authenticate(username, password):
    df_users = pd.read_csv("users.csv").astype(str) if Path("users.csv").exists() else pd.DataFrame()
    hashed_pw = hash_password(password)
    row = df_users[(df_users.username == username) & (df_users.password == hashed_pw)]
    return {"username": username, "role": row.iloc[0].role} if not row.empty else None

def create_access_token(data, expires=None):
    exp = datetime.utcnow() + (expires or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return jwt.encode({**data, "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token):
    try: return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError: return None

def safe_translate(txt, src="vi", tgt="en"):
    try: return GoogleTranslator(source=src, target=tgt).translate(txt)
    except: return txt

def get_qwen3_embedding(text):
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = embed_model(**inputs)
        embedding = output.last_hidden_state[:, -1, :]  # lấy token cuối
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding[0].cpu().numpy()


def find_top_doc(question):
    q_vec = get_qwen3_embedding(question)
    sims = cosine_similarity([q_vec], df_chunks_128[embedding_cols].values.astype("float32"))[0]
    best = df_chunks_128.assign(similarity=sims).nlargest(1, "similarity").iloc[0]
    return best.doc_id, q_vec

def find_related_file(question_vi):
    question_vi_lower = question_vi.lower()
    doc_id, _ = find_top_doc(question_vi_lower)
    df_docs = pd.read_csv("documents.csv") if Path("documents.csv").exists() else pd.DataFrame()
    file_info = df_docs[df_docs.doc_id == doc_id].iloc[0]
    return {"file_name": file_info.file_name, "file_url": f"/download/{file_info.file_path}"}

def find_chunks_and_answer(question_vi, top_k=1):
    question_vi_lower = question_vi.lower()
    doc_id, q_vec = find_top_doc(question_vi_lower)
    df_docs = pd.read_csv("documents.csv") if Path("documents.csv").exists() else pd.DataFrame()
    file_info = df_docs[df_docs.doc_id == doc_id].iloc[0]
    chunks = df_chunks_128[df_chunks_128.doc_id == doc_id].copy()
    chunks["similarity"] = cosine_similarity([q_vec], chunks[embedding_cols].values.astype("float32"))[0]
    top_idx = chunks.sort_values("similarity", ascending=False).iloc[0].chunk_idx
    context_en = "\n\n".join(
        safe_translate(c.chunk_text_vi, "vi", "en") if pd.isna(c.get("chunk_text_en")) else c.chunk_text_en
        for _, c in chunks[chunks.chunk_idx >= top_idx].nsmallest(top_k+2, "chunk_idx").iterrows()
    )
    question_en = safe_translate(question_vi, "vi", "en")
    prompt = f"1. Use only the context below to answer.\n\nContext:\n{context_en}\n\nQuestion: {question_en}\nAnswer:\n"
    ids = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    with torch.inference_mode():
        output = llm_model.generate(**ids, max_new_tokens=100, pad_token_id=llm_tokenizer.eos_token_id)
    answer_en = llm_tokenizer.decode(output[0], skip_special_tokens=True).split("Answer:")[-1].strip()
    return {"answer_vi": safe_translate(answer_en, "en", "vi"), "file_name": file_info.file_name,
            "file_url": f"/download/{file_info.file_path}"}

# --- Chunk Pipeline ---
def run_chunk_pipeline(file_path, doc_id):
    os.makedirs("chunk_viet", exist_ok=True)
    output_file = "chunk_viet/chunk_viet_256_128_translated.csv"
    translated_file = "chunk_viet/chunk_viet_256_128_translated.csv"

    def clean_text_basic(text):
        text = re.sub(r"[.]{4,}|[…]{2,}", "...", text).replace("…", "...")
        for bullet in ["•", "▪", "◦", "·", "●"]: text = text.replace(bullet, "-")
        for space_char in ["\u200b", "\u200c", "\u200d", "\xa0"]: text = text.replace(space_char, " ")
        return text

    def clean_text_after_split(text):
        text = text.replace("\r\n", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
        return "\n".join([line for line in lines if line])

    def normalize_text_oneline(text): return re.sub(r"\s+", " ", text).strip()

    def split_text_by_token_limit(text, max_tokens=128, overlap=0):
        tokens = embed_tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
        return [embed_tokenizer.decode(tokens[i:i+max_tokens], skip_special_tokens=True)
                for i in range(0, len(tokens), max_tokens - overlap)]

    def batch_get_embeddings(text_list, batch_size=8):
        embs = []
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i+batch_size]
            inputs = embed_tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            with torch.no_grad():
                output = embed_model(**inputs)
                embedding = output.last_hidden_state[:, -1, :]  # lấy token cuối
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                embs.extend(embedding.cpu().numpy())
        return embs

    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".doc": elements = partition_doc(filename=file_path)
    elif ext == ".docx": elements = partition_docx(filename=file_path)
    elif ext == ".pdf": elements = partition_pdf(filename=file_path)
    else: raise Exception(f"Định dạng không hỗ trợ: {file_path}")

    raw_text = "\n".join([e.text for e in elements if hasattr(e, "text") and isinstance(e.text, str) and e.text.strip()])
    segments = re.split(r"\n{3,}", clean_text_basic(raw_text))

    chunks_vi, chunks_ori = [], []
    for segment in segments:
        cleaned = clean_text_after_split(segment)
        if not cleaned.strip(): continue
        for ch in split_text_by_token_limit(cleaned): 
            chunks_ori.append(ch); chunks_vi.append(normalize_text_oneline(ch))

    embeddings = batch_get_embeddings(chunks_vi)
    chunk_id_start = 1
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        df_prev = pd.read_csv(output_file)
        if not df_prev.empty: chunk_id_start = df_prev["chunk_id"].max() + 1

    header = ["chunk_id", "doc_id", "chunk_idx", "chunk_text_vi", "chunk_text_ori"] + [str(i) for i in range(embed_dim)]
    write_header = not os.path.exists(output_file) or os.path.getsize(output_file) == 0
    with open(output_file, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        if write_header: writer.writerow(header)
        for idx, (vi, ori, emb) in enumerate(zip(chunks_vi, chunks_ori, embeddings)):
            writer.writerow([chunk_id_start + idx, doc_id, idx, vi, ori] + list(emb))

    df_doc = pd.DataFrame({"chunk_text_vi": chunks_vi})
    REMOVE_PATTERNS = [
        "TRƯỜNG ĐẠI HỌC THỦY LỢI","TRƯỜNG ĐH THỦY LỢI","PHÒNG ĐÀO TẠO",
        "CỘNG HOÀ XÃ HỘI CHỦ NGHĨA VIỆT NAM","CỘNG HOÀ XHCN VIỆT NAM",
        "Độc lập - Tự do - Hạnh phúc","Độc lập – Tự do – Hạnh phúc","BỘ NÔNG NGHIỆP VÀ PTNT"
    ]
    pat = re.compile("|".join(map(re.escape, REMOVE_PATTERNS)), flags=re.IGNORECASE)
    df_doc["chunk_text_vi_cleaned"] = df_doc["chunk_text_vi"].apply(lambda t: pat.sub("", t).strip())
    df_doc["chunk_text_en"] = df_doc["chunk_text_vi_cleaned"].apply(lambda t: safe_translate(t, "vi", "en"))
    df_doc["doc_id"] = doc_id
    df_translated = pd.read_csv(translated_file) if os.path.exists(translated_file) else pd.DataFrame()
    df_translated = pd.concat([df_translated, df_doc], ignore_index=True)
    df_translated.to_csv(translated_file, index=False)

def stream_chunks_and_answer(question_vi, top_k=1):
    """
    Stream câu trả lời từng phần (token-by-token) từ LLM.
    """
    question_vi_lower = question_vi.lower()
    doc_id, q_vec = find_top_doc(question_vi_lower)
    df_docs = pd.read_csv("documents.csv") if Path("documents.csv").exists() else pd.DataFrame()
    file_info = df_docs[df_docs.doc_id == doc_id].iloc[0]

    # --- Tìm context ---
    chunks = df_chunks_128[df_chunks_128.doc_id == doc_id].copy()
    chunks["similarity"] = cosine_similarity([q_vec], chunks[embedding_cols].values.astype("float32"))[0]
    top_idx = chunks.sort_values("similarity", ascending=False).iloc[0].chunk_idx
    context_en = "\n\n".join(
        safe_translate(c.chunk_text_vi, "vi", "en") if pd.isna(c.get("chunk_text_en")) else c.chunk_text_en
        for _, c in chunks[chunks.chunk_idx >= top_idx].nsmallest(top_k + 2, "chunk_idx").iterrows()
    )

    # --- Tạo prompt ---
    question_en = safe_translate(question_vi, "vi", "en")
    prompt = f"1. Use only the context below to answer.\n\nContext:\n{context_en}\n\nQuestion: {question_en}\nAnswer:\n"

    # --- Token streamer ---
    streamer = TextIteratorStreamer(llm_tokenizer, skip_prompt=True)
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=200,
        streamer=streamer,
        pad_token_id=llm_tokenizer.eos_token_id
    )

    thread = Thread(target=llm_model.generate, kwargs=generation_kwargs)
    thread.start()

    # --- Stream từng token ---
    for new_text in streamer:
        yield new_text