from pydantic import BaseModel, EmailStr

from app.schemas.user import UserProfile

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserProfile