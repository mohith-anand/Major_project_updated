from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt, JWTError
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
import os

from . import models
from .database import SessionLocal


# ===============================
# PASSWORD HASHING (ARGON2 + BCRYPT)
# ===============================

pwd_context = CryptContext(
    schemes=["argon2", "bcrypt"],   # ✅ Supports both
    deprecated="auto"
)

SECRET = os.getenv("SECRET_KEY")
if not SECRET:
    raise RuntimeError("SECRET_KEY environment variable not set. Application cannot start.")

ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

security = HTTPBearer()


# ===============================
# DATABASE DEPENDENCY
# ===============================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ===============================
# PASSWORD FUNCTIONS
# ===============================

def hash_password(password: str):
    if not isinstance(password, str):
        raise ValueError("Password must be string")
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str):
    return pwd_context.verify(plain, hashed)


# ===============================
# JWT CREATION
# ===============================

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET, algorithm=ALGORITHM)


# ===============================
# GET CURRENT USER (DB OBJECT)
# ===============================

def get_current_user(
    token: HTTPAuthorizationCredentials = Security(security),
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Invalid authentication credentials",
    )

    try:
        payload = jwt.decode(
            token.credentials,
            SECRET,
            algorithms=[ALGORITHM]
        )

        username: str = payload.get("sub")

        if username is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    user = db.query(models.User).filter(
        models.User.username == username
    ).first()

    if user is None:
        raise credentials_exception

    return user