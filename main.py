from fastapi import FastAPI, HTTPException, UploadFile, Form, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pathlib import Path
import pandas as pd
import mimetypes
from server.models import LoginRequest, UserCreate, UserUpdate, UserRegister
from server.utils import (
    normalize_filename, save_csv, next_id, get_abs_path, authenticate,
    create_access_token, verify_token, find_chunks_and_answer,
    find_related_file, run_chunk_pipeline, register_user
)
from fastapi.responses import StreamingResponse
from server.utils import stream_chunks_and_answer
# ====================== INIT ======================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "data"
UPLOAD_DIR.mkdir(exist_ok=True)

USER_CSV = "users.csv"
DOC_CSV = "documents.csv"
ISCHUNK_CSV = "ischunk.csv"

# --- Load Data ---
df_users = pd.read_csv(USER_CSV, dtype={"id": int, "username": str, "role": str, "password": str}) \
    if Path(USER_CSV).exists() else pd.DataFrame(columns=["id", "username", "role", "password"])

df_docs = pd.read_csv(DOC_CSV) if Path(DOC_CSV).exists() else pd.DataFrame(
    columns=["doc_id", "file_name", "summary", "file_path", "upload_date"]
)

df_ischunk = pd.read_csv(ISCHUNK_CSV) if Path(ISCHUNK_CSV).exists() else pd.DataFrame(
    columns=["doc_id", "file_name", "is_chunked"]
)

try:
    df_chunks_128
except NameError:
    df_chunks_128 = pd.DataFrame()

# --- Validation Error Handler ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors(), "body": exc.body})

# ====================== AUTH ======================
@app.post("/register")
def register(data: UserRegister):
    return register_user(data.username, data.password)

@app.post("/login")
def login(data: LoginRequest):
    user = authenticate(data.username, data.password)
    if not user:
        raise HTTPException(401, "Sai tài khoản hoặc mật khẩu")
    return {"access_token": create_access_token(user), **user}

@app.get("/verify-token")
def verify_token_endpoint(token: str):
    payload = verify_token(token)
    if not payload:
        raise HTTPException(401, "Token không hợp lệ hoặc hết hạn")
    return {"valid": True, "user": payload}

# ====================== ASK ======================
@app.get("/ask")
def ask(question: str):
    return JSONResponse(find_chunks_and_answer(question))

@app.get("/ask_file")
def ask_file(question: str):
    return JSONResponse(find_related_file(question))

@app.get("/ask_stream")
def ask_stream(question: str):
    """
    Stream câu trả lời từng phần.
    """
    return StreamingResponse(stream_chunks_and_answer(question), media_type="text/plain")
# ====================== FILE ======================
@app.get("/download/{file_path}")
def download_file(file_path: str):
    return FileResponse(get_abs_path(file_path))

@app.get("/preview/{file_path}")
def preview_file(file_path: str):
    path = get_abs_path(file_path)
    return FileResponse(path, media_type=mimetypes.guess_type(path)[0] or "application/octet-stream")

# ====================== DOCUMENTS ======================
@app.get("/documents")
def get_documents():
    return df_docs.to_dict(orient="records")

@app.post("/documents/upload")
async def upload_document(file_name: str = Form(...), summary: str = Form(""), file: UploadFile = None):
    global df_docs, df_ischunk
    if not file:
        raise HTTPException(400, "Thiếu file upload")
    doc_id = next_id(df_docs, "doc_id")
    safe_filename = normalize_filename(file.filename)
    save_path = UPLOAD_DIR / safe_filename
    save_path.write_bytes(await file.read())
    df_docs = pd.concat([df_docs, pd.DataFrame([{
        "doc_id": doc_id,
        "file_name": normalize_filename(file_name) or safe_filename,
        "summary": summary,
        "file_path": safe_filename,
        "upload_date": pd.Timestamp.now().strftime("%Y-%m-%d")
    }])], ignore_index=True)
    save_csv(df_docs, DOC_CSV)

    df_ischunk = pd.concat([df_ischunk, pd.DataFrame([{
        "doc_id": doc_id, "file_name": file_name, "is_chunked": 0
    }])], ignore_index=True)
    save_csv(df_ischunk, ISCHUNK_CSV)

    return {"message": "Đã thêm tài liệu", "doc_id": doc_id}

@app.put("/documents/{doc_id}")
async def update_document(doc_id: int, file_name: str = Form(...), summary: str = Form(""), file: UploadFile = File(None)):
    global df_docs
    idx = df_docs.index[df_docs["doc_id"] == doc_id]
    if len(idx) == 0:
        raise HTTPException(404, "Không tìm thấy tài liệu")

    # Nếu có file mới thì ghi đè
    if file:
        safe_filename = normalize_filename(file.filename)
        save_path = UPLOAD_DIR / safe_filename
        save_path.write_bytes(await file.read())
        df_docs.loc[idx[0], "file_path"] = safe_filename

    df_docs.loc[idx[0], ["file_name", "summary"]] = [file_name, summary]
    save_csv(df_docs, DOC_CSV)
    return {"message": "Đã cập nhật tài liệu", "doc_id": doc_id}

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int):
    global df_docs, df_ischunk, df_chunks_128
    row = df_docs[df_docs.doc_id == doc_id]
    if row.empty:
        raise HTTPException(404, "Không tìm thấy tài liệu")

    path = UPLOAD_DIR / normalize_filename(row.iloc[0].file_path)
    if path.exists():
        path.unlink()

    df_docs = df_docs[df_docs.doc_id != doc_id]
    df_ischunk = df_ischunk[df_ischunk.doc_id != doc_id]
    save_csv(df_docs, DOC_CSV)
    save_csv(df_ischunk, ISCHUNK_CSV)

    translated_file = "chunk_viet/chunk_viet_256_128_translated.csv"
    if Path(translated_file).exists():
        df_translated = pd.read_csv(translated_file)
        df_translated = df_translated[df_translated["doc_id"] != doc_id]
        df_translated.to_csv(translated_file, index=False)

    if not df_chunks_128.empty:
        df_chunks_128 = df_chunks_128[df_chunks_128.doc_id != doc_id]

    return {"message": "Đã xóa tài liệu, ischunk và chunk liên quan"}

# ====================== USERS ======================
@app.get("/users")
def get_users():
    df_users = pd.read_csv("users.csv") if Path("users.csv").exists() else pd.DataFrame(
        columns=["id", "username", "role", "password"]
    )
    return df_users.to_dict(orient="records")

@app.post("/users")
def add_user(user: UserCreate):
    global df_users
    user_id = next_id(df_users, "id")
    new_user = {"id": user_id, **user.dict()}
    df_users = pd.concat([df_users, pd.DataFrame([new_user])], ignore_index=True)
    save_csv(df_users, USER_CSV)
    return {"message": "Đã thêm người dùng", "id": user_id}

@app.put("/users/{user_id}")
def update_user(user_id: int, user: UserUpdate):
    global df_users
    idx = df_users.index[df_users["id"] == user_id]
    if len(idx) == 0:
        raise HTTPException(404, "Không tìm thấy người dùng")
    df_users.loc[idx[0], ["username", "role"]] = [user.username, user.role]
    save_csv(df_users, USER_CSV)
    return {"message": "Đã cập nhật người dùng"}

@app.delete("/users/{user_id}")
def delete_user(user_id: int):
    global df_users
    before_count = len(df_users)
    df_users = df_users[df_users["id"] != user_id]
    save_csv(df_users, USER_CSV)
    if len(df_users) == before_count:
        raise HTTPException(404, f"Không tìm thấy user ID {user_id}")
    return {"message": f"Đã xóa người dùng {user_id}"}

# ====================== CHUNK ======================
@app.get("/chunks/status")
def get_chunk_status():
    global df_docs, df_ischunk
    for _, row in df_docs.iterrows():
        if not (df_ischunk["doc_id"] == row["doc_id"]).any():
            df_ischunk = pd.concat([df_ischunk, pd.DataFrame([{
                "doc_id": row["doc_id"], "file_name": row["file_name"], "is_chunked": 0
            }])], ignore_index=True)
    save_csv(df_ischunk, ISCHUNK_CSV)
    return df_ischunk.to_dict(orient="records")

@app.post("/chunks/run/{doc_id}")
def chunk_file(doc_id: int):
    global df_docs, df_ischunk
    row = df_docs[df_docs.doc_id == doc_id]
    if row.empty:
        raise HTTPException(404, "Không tìm thấy tài liệu")
    file_path = UPLOAD_DIR / row.iloc[0]["file_path"]
    try:
        run_chunk_pipeline(str(file_path), doc_id)
        df_ischunk.loc[df_ischunk.doc_id == doc_id, "is_chunked"] = 1
        save_csv(df_ischunk, ISCHUNK_CSV)
        return {"message": f"Đã chunk xong file {file_path}"}
    except Exception as e:
        raise HTTPException(500, f"Lỗi khi chunk: {e}")

