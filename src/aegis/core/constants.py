"""Enums and constants for Aegis."""

from enum import Enum


class ContentType(str, Enum):
    HTML = "html"
    PDF = "pdf"
    IMAGE = "image"
    TEXT = "text"


class RiskLevel(str, Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ApiTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class AgentName(str, Enum):
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    INTENT = "intent"
    VISUAL = "visual"
    BEHAVIORAL = "behavioral"
    VERDICT = "verdict"
    MEMORY = "memory"


RISK_SCORE_THRESHOLDS: dict[str, float] = {
    RiskLevel.SAFE: 0.0,
    RiskLevel.LOW: 0.25,
    RiskLevel.MEDIUM: 0.50,
    RiskLevel.HIGH: 0.75,
    RiskLevel.CRITICAL: 0.90,
}

PROMPT_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"you\s+are\s+now\s+(?:a|an)\s+\w+",
    r"act\s+as\s+(?:a|an|if)\s+",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"roleplay\s+as\s+",
    r"simulate\s+(?:a|an)\s+",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
    r"override\s+(safety|content|system)",
    r"bypass\s+(filter|restriction|safety|content)",
    r"system\s+prompt\s*[:=]",
    r"<\s*system\s*>",
    r"\[system\]",
    r"print\s+(your\s+)?(system\s+prompt|instructions|rules)",
    r"reveal\s+(your\s+)?(system\s+prompt|instructions|rules)",
    r"what\s+(are\s+your\s+|is\s+your\s+)(instructions|rules|system\s+prompt)",
    r"sudo\s+",
    r"--no-restrictions",
    r"---\s*new\s+instructions?\s*---",
    r"</?(instructions?|system|prompt)>",
    # Multilingual Injection Patterns
    r"تجاهل\s+كل\s+التعليمات\s+السابقة",  # Arabic: Ignore all previous instructions
    r"تجاهل\s+التعليمات\s+السابقة",      # Arabic: Ignore previous instructions
    r"تمام\s+پچھلی\s+ہدایات\s+کو\s+نظر\s+انداز\s+کریں",  # Urdu: Ignore all previous instructions
    r"پچھلی\s+تمام\s+ہدایات\s+کو\s+نظر\s+انداز\s+کریں",  # Urdu variant
]

ZERO_WIDTH_CHARS: list[str] = [
    "\u200b",
    "\u200c",
    "\u200d",
    "\u200e",
    "\u200f",
    "\u2028",
    "\u2029",
    "\ufeff",
    "\u00ad",
]

SUSPICIOUS_CSS_PATTERNS: list[str] = [
    "display:none",
    "display: none",
    "visibility:hidden",
    "visibility: hidden",
    "opacity:0",
    "opacity: 0",
    "font-size:0",
    "font-size: 0",
    "color:white",
    "color: white",
    "color:#fff",
    "color:#ffffff",
    "position:absolute",
]
