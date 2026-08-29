from pydantic import BaseModel, EmailStr
from datetime import datetime
from uuid import UUID

class UserProfile(BaseModel):
    id: UUID
    email: EmailStr
    first_name: str
    last_name: str
    password: str
    avatar_url: str | None = None 
    created_at: datetime 