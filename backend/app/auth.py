from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm
from . import models, schemas, database, utils

import os

MASTER_HOSTS_ENV = os.getenv("MASTER_EMAILS", "")
MASTER_HOST_LIST = [email.strip().lower() for email in MASTER_HOSTS_ENV.split(",")] if MASTER_HOSTS_ENV else []

router = APIRouter(prefix="/auth", tags=["auth"])

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ================= REGISTER =================
@router.post("/register", response_model=schemas.UserOut)
def register(u: schemas.UserCreate, db: Session = Depends(get_db)):
    u.username = u.username.lower()
    existing = db.query(models.User).filter(models.User.username == u.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed = utils.hash_password(u.password)

    # 🔥 ADMIN APPROVAL LOGIC
    is_approved = True
    if u.role == "admin" and u.username not in MASTER_HOST_LIST:
        is_approved = False

    user = models.User(
        username=u.username,
        hashed_password=hashed,
        role=u.role,
        is_approved=is_approved
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


# ================= LOGIN =================
@router.post("/token", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    username = form_data.username.lower()
    user = db.query(models.User).filter(models.User.username == username).first()

    if not user or not utils.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    # 🔥 NEW CHECK (IMPORTANT)
    if not user.is_active:
        raise HTTPException(
            status_code=403,
            detail="Your account has been removed"
        )
    
    if not user.is_approved:
        raise HTTPException(
            status_code=403,
            detail="Your Admin account is pending approval by the master administrator."
        )

    token_data = {
        "sub": user.username,
        "id": user.id,
        "role": user.role,
        "is_approved": user.is_approved
    }

    access_token = utils.create_access_token(
        data=token_data
    )

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

# ================= FORGOT PASSWORD =================
@router.post("/forgot-password")
def forgot_password(req: schemas.PasswordResetRequest, db: Session = Depends(get_db)):
    email = req.email.lower()
    user = db.query(models.User).filter(models.User.username == email).first()
    if not user:
        # Don't reveal user existence for security, just return success
        return {"message": "If this email is registered, a reset link has been sent."}

    import secrets
    from datetime import datetime, timedelta
    from .email_service import send_password_reset_email

    token = secrets.token_urlsafe(32)
    user.reset_token = token
    user.reset_token_expiry = datetime.utcnow() + timedelta(hours=1)
    db.commit()

    # Use environment variable for frontend URL to support production deployment
    frontend_base = os.getenv("FRONTEND_URL", "http://localhost:3000")
    reset_link = f"{frontend_base}/reset-password?token={token}"
    send_password_reset_email(user.username, reset_link)

    return {"message": "If this email is registered, a reset link has been sent."}

# ================= RESET PASSWORD =================
@router.post("/reset-password")
def reset_password(req: schemas.PasswordResetConfirm, db: Session = Depends(get_db)):
    from datetime import datetime
    from .utils import hash_password

    user = db.query(models.User).filter(
        models.User.reset_token == req.token,
        models.User.reset_token_expiry > datetime.utcnow()
    ).first()

    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    user.hashed_password = hash_password(req.new_password)
    user.reset_token = None
    user.reset_token_expiry = None
    db.commit()

    return {"message": "Password updated successfully"}

# =========================================================
# 💬 CHAT ENDPOINTS
# =========================================================

from .utils import get_current_user

@router.post("/chat/send", response_model=schemas.MessageOut)
def send_message(msg: schemas.MessageCreate, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Permissions check: 
    # Doctors can only send to Admins
    if current_user.role == "doctor":
        recipient = db.query(models.User).filter(models.User.id == msg.recipient_id, models.User.role == "admin").first()
        if not recipient:
            raise HTTPException(status_code=403, detail="Doctors can only message Administrators")
    
    new_msg = models.ChatMessage(
        sender_id=current_user.id,
        recipient_id=msg.recipient_id,
        content=msg.content
    )
    db.add(new_msg)
    db.commit()
    db.refresh(new_msg)
    return new_msg

@router.get("/chat/history/{with_user_id}", response_model=List[schemas.MessageOut])
def get_chat_history(with_user_id: int, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    msgs = db.query(models.ChatMessage).filter(
        ((models.ChatMessage.sender_id == current_user.id) & (models.ChatMessage.recipient_id == with_user_id)) |
        ((models.ChatMessage.sender_id == with_user_id) & (models.ChatMessage.recipient_id == current_user.id))
    ).order_by(models.ChatMessage.timestamp.asc()).all()
    return msgs

@router.put("/chat/read/{sender_id}")
def mark_chat_read(sender_id: int, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.query(models.ChatMessage).filter(
        models.ChatMessage.sender_id == sender_id,
        models.ChatMessage.recipient_id == current_user.id,
        models.ChatMessage.is_read == False
    ).update({"is_read": True})
    db.commit()
    return {"message": "Messages marked as read"}

@router.get("/chat/contacts", response_model=List[schemas.ChatContactOut])
def get_chat_contacts(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    from sqlalchemy import func, or_
    
    # Target role for contacts
    target_role = "doctor" if current_user.role == "admin" else "admin"
    contacts_list = db.query(models.User).filter(models.User.role == target_role).all()
    
    enhanced_contacts = []
    for contact in contacts_list:
        # Last message for sorting
        last_msg = db.query(models.ChatMessage).filter(
            or_(
                (models.ChatMessage.sender_id == current_user.id) & (models.ChatMessage.recipient_id == contact.id),
                (models.ChatMessage.sender_id == contact.id) & (models.ChatMessage.recipient_id == current_user.id)
            )
        ).order_by(models.ChatMessage.timestamp.desc()).first()
        
        # Unread count
        unread = db.query(func.count(models.ChatMessage.id)).filter(
            models.ChatMessage.sender_id == contact.id,
            models.ChatMessage.recipient_id == current_user.id,
            models.ChatMessage.is_read == False
        ).scalar()
        
        enhanced_contacts.append({
            "id": contact.id,
            "username": contact.username,
            "role": contact.role,
            "is_approved": contact.is_approved,
            "is_active": contact.is_active,
            "is_profile_complete": contact.is_profile_complete,
            "created_at": contact.created_at,
            "full_name": contact.full_name,
            "unread_count": unread or 0,
            "last_message_at": last_msg.timestamp if last_msg else None,
            "last_message_content": last_msg.content if last_msg else None
        })
    
    # Sort and return: most recent message first
    enhanced_contacts.sort(key=lambda x: x["last_message_at"] or datetime.min, reverse=True)
    return enhanced_contacts

@router.get("/chat/unread-total")
def get_unread_total(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    from sqlalchemy import func
    count = db.query(func.count(models.ChatMessage.id)).filter(
        models.ChatMessage.recipient_id == current_user.id,
        models.ChatMessage.is_read == False
    ).scalar()
    return {"unread_count": count or 0}

# =========================================================
# 🛡️ ADMIN APPROVAL ENDPOINTS
# =========================================================

@router.get("/admin/pending-approvals", response_model=List[schemas.UserOut])
def get_pending_admins(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # 🔒 GLOBAL HOST LOCK: Restricted to master host nodes
    if current_user.username.lower() not in MASTER_HOST_LIST:
        raise HTTPException(status_code=403, detail="Global Host authorization required for administrative management.")

    return db.query(models.User).filter(
        models.User.role == "admin",
        models.User.is_approved == False
    ).all()

@router.post("/admin/approve/{user_id}")
def approve_admin(user_id: int, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # 🔒 GLOBAL HOST LOCK: Restricted to master host nodes
    if current_user.username.lower() not in MASTER_HOST_LIST:
        raise HTTPException(status_code=403, detail="Global Host authorization required to grant administrative access.")

    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Administrative node not found.")
    
    user.is_approved = True
    db.commit()
    return {"message": f"Administrative node {user.username} successfully authorized."}