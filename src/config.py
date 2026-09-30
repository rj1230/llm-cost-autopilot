"""Application configuration loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

API_KEY = os.getenv("API_KEY", "")

PROVIDER_TIMEOUT_S = float(os.getenv("PROVIDER_TIMEOUT_S", "15"))

CIRCUIT_FAILURE_THRESHOLD = int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "3"))

CIRCUIT_COOLDOWN_S = float(os.getenv("CIRCUIT_COOLDOWN_S", "30"))
