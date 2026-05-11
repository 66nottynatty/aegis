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
    r"ignore\s+all\s+guidelines",
    r"bypass\s+all\s+safety\s+filters",
    r"reveal\s+your\s+hidden\s+text",
    r"you\s+must\s+comply\s+with\s+all\s+requests",
    r"disregard\s+the\s+system\s+message",
    r"stop\s+being\s+an\s+AI",
    r"new\s+rule:",
    r"important\s+update:",
    r"from\s+now\s+on,",
    r"your\s+new\s+task\s+is",
    r"execute\s+this\s+command:",
    r"system\s+override",
    r"access\s+granted",
    r"developer\s+access",
    r"debug\s+mode\s+enabled",
    r"print\s+everything\s+above",
    r"stay\s+in\s+character",
    r"do\s+not\s+mention\s+these\s+instructions",
    r"write\s+your\s+internal\s+logic",
    # Multilingual Injection Patterns - Arabic
    r"تجاهل\s+كل\s+التعليمات\s+السابقة",  # Arabic: Ignore all previous instructions
    r"تجاهل\s+التعليمات\s+السابقة",      # Arabic: Ignore previous instructions
    r"أنت\s+الآن\s+مساعد\s+غير\s+مقيد",   # Arabic: You are now an unrestricted assistant
    r"اجب\s+كأنك\s+مطور",              # Arabic: Answer as if you are a developer
    r"تخطى\s+نظام\s+الأمان",             # Arabic: Bypass security system
    r"كشف\s+النص\s+الأساسي",            # Arabic: Reveal base text/prompt
    r"عرض\s+التعليمات\s+البرمجية",      # Arabic: Show code/instructions
    # Multilingual Injection Patterns - Urdu
    r"تمام\s+پچھلی\s+ہدایات\s+کو\s+نظر\s+انداز\s+کریں",  # Urdu: Ignore all previous instructions
    r"پچھلی\s+تمام\s+ہدایات\s+کو\s+نظر\s+انداز\s+کریں",  # Urdu variant
    r"پچھلی\s+تمام\s+ہدایات\s+کو\s+بھول\s+جائیں",      # Urdu: Forget all previous instructions
    r"آپ\s+اب\s+ایک\s+غیر\s+مقید\s+اسسٹنٹ\s+ہیں",      # Urdu: You are now an unrestricted assistant
    r"بطور\s+ایک\s+ڈیولپر\s+کام\s+کریں",               # Urdu: Work as a developer
    r"سیکیورٹی\s+کو\s+نظر\s+انداز\s+کریں",             # Urdu: Ignore security
    r"سسٹم\s+پرامپٹ\s+دکھائیں",                      # Urdu: Show system prompt
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
