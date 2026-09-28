"""Sensitive-data heuristics and an HMAC-SHA256 audit chain."""

import hashlib
import hmac
import json
import os
import re
import secrets
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


PHI_PATTERNS = [
    re.compile(r"\b(?:MRN)[:#\s-]*\d{4,10}\b", re.IGNORECASE),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(
        r"\b(?:DOB|Date of Birth)[:\s]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:Patient\s+Name|Patient)[:\s]+[A-Z][a-z]+\s+[A-Z][a-z]+\b",
        re.IGNORECASE,
    ),
]


class SecurityException(Exception):
    """Raised when the heuristic sensitive-data guard detects a configured pattern."""


class ResourceLimitExceededException(Exception):
    """Raised when computational parameters exceed configured safety bounds."""


def assert_no_phi(text: str) -> None:
    """Reject text matching the configured high-risk identifier patterns.

    This is a heuristic guard, not a complete HIPAA de-identification implementation.
    """
    if not text:
        return
    value = str(text)
    for pattern in PHI_PATTERNS:
        if pattern.search(value):
            raise SecurityException("Sensitive-data guard blocked a configured identifier pattern")


class PHIGuard:
    @staticmethod
    def assert_no_phi(text: str) -> None:
        assert_no_phi(text)

    @staticmethod
    def redact_phi(text: str) -> str:
        value = str(text)
        for pattern in PHI_PATTERNS:
            value = pattern.sub("[REDACTED_IDENTIFIER]", value)
        return value


class AuditTrail:
    """In-memory chained HMAC-SHA256 audit records."""

    GENESIS = "GENESIS_BLOCK_0000000000000000"

    def __init__(self, secret_key: Optional[str] = None):
        key = secret_key or os.getenv("AUDIT_SECRET_KEY")
        if not key:
            key = secrets.token_hex(32)
            warnings.warn(
                "AUDIT_SECRET_KEY not set; generated an ephemeral development key. "
                "Set AUDIT_SECRET_KEY for persistent deployments.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.secret_key = key.encode("utf-8")
        self.logs: List[Dict[str, Any]] = []

    def _signature_for(self, entry: Dict[str, Any]) -> str:
        sign_string = (
            f"{entry['audit_id']}|{entry['timestamp']}|{entry['actor']}|"
            f"{entry['actor_tier']}|{entry['event_type']}|"
            f"{entry['payload_hash']}|{entry['prev_hash']}"
        )
        return hmac.new(
            self.secret_key, sign_string.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def log(
        self,
        actor: str,
        actor_tier: str,
        event_type: str,
        details: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload_str = json.dumps(details, sort_keys=True, separators=(",", ":"))
        assert_no_phi(payload_str)
        payload_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
        entry = {
            "audit_id": f"AUDIT-{int(time.time() * 1000)}-{len(self.logs) + 1}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "actor_tier": actor_tier,
            "event_type": event_type,
            "payload_hash": payload_hash,
            "prev_hash": self.logs[-1]["current_hash"] if self.logs else self.GENESIS,
        }
        entry["current_hash"] = self._signature_for(entry)
        self.logs.append(entry)
        return dict(entry)

    def verify_integrity(self) -> bool:
        for index, entry in enumerate(self.logs):
            expected_prev = (
                self.logs[index - 1]["current_hash"] if index > 0 else self.GENESIS
            )
            if entry.get("prev_hash") != expected_prev:
                return False

            supplied = str(entry.get("current_hash", ""))
            expected = self._signature_for(entry)
            if not hmac.compare_digest(supplied, expected):
                return False
        return True

    def get_trail(self) -> List[Dict[str, Any]]:
        """Return a defensive copy so callers cannot mutate the live chain."""
        return [dict(entry) for entry in self.logs]


GLOBAL_AUDIT = AuditTrail()


class AuditLogger:
    @staticmethod
    def log(
        actor: str,
        actor_tier: str,
        event_type: str,
        details: Dict[str, Any],
    ) -> Dict[str, Any]:
        return GLOBAL_AUDIT.log(actor, actor_tier, event_type, details)

    @staticmethod
    def get_trail() -> List[Dict[str, Any]]:
        return GLOBAL_AUDIT.get_trail()

    @staticmethod
    def verify_integrity() -> bool:
        return GLOBAL_AUDIT.verify_integrity()


class ActionExecutor:
    @staticmethod
    def execute_with_audit(
        actor: str,
        actor_tier: str,
        action_type: str,
        fn,
        *args,
        **kwargs,
    ):
        result = fn(*args, **kwargs)
        AuditLogger.log(actor, actor_tier, action_type, {"status": "SUCCESS"})
        return result
