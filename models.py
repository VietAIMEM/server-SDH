from pydantic import BaseModel
from typing import Optional

class LoginRequest(BaseModel):
    username: str
    password: str

class Document(BaseModel):
    doc_id: int
    file_name: str
    summary: str
    file_path: str
    upload_date: str

class DocumentUpdate(BaseModel):
    file_name: str
    summary: Optional[str] = None

class UserItem(BaseModel):
    id: int
    username: str
    role: str
    password: Optional[str] = None

class UserCreate(BaseModel):
    username: str
    role: str
    password: str

class UserUpdate(BaseModel):
    username: str
    role: str


class UserRegister(BaseModel):
    username: str
    password: str