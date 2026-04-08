from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class PatientMessage(Base):
    __tablename__ = "patient_messages"

    id = Column(Integer, primary_key=True, index=True)
    whatsapp_id = Column(String(128), nullable=False, index=True)
    phone_number = Column(String(32), nullable=False, index=True)
    message_text = Column(Text, nullable=False)
    language = Column(String(32), nullable=True)
    message_type = Column(String(32), nullable=False, default="text")
    created_at = Column(DateTime, default=datetime.utcnow)
