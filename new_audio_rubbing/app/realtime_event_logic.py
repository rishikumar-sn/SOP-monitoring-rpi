from __future__ import annotations

import time

JEWEL_CLASS = "JEWEL_RUB_OK"
STRONG_NEGATIVE_CLASSES = {
    "NAIL_RUB_NOK",
    "FINGER_RUB_NOK",
    "STONE_TAP_HANDLING_NOK",
}


class RubbingEventLogic:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.event_active: bool = False
        self.event_start_time: float | None = None
        self.event_locked_nok: bool = False
        self.event_confirmed_ok: bool = False
        self.consecutive_jewel_hits: int = 0
        self.early_negative_hits: int = 0
        self._silence_start_time: float | None = None

    def update(
        self,
        predicted_class: str,
        confidence: float,
        rms_energy: float,
        visual_jewel_detected: bool,
        embedding_verified: bool,
        ok_confidence: float = 0.85,
        negative_confidence: float = 0.70,
        silence_reset_sec: float = 0.5,
        silence_threshold: float = 0.005,
        required_jewel_hits: int = 3,
        negative_lock_hits: int = 3,
    ) -> dict[str, object]:
        now = time.monotonic()

        if not visual_jewel_detected:
            self.reset()
            return self._result("NOK", "visual_jewel_not_detected", rms_energy)

        if rms_energy < silence_threshold:
            if self._silence_start_time is None:
                self._silence_start_time = now
            elif now - self._silence_start_time >= silence_reset_sec:
                self.reset()
                return self._result("NOK", "silence_reset", rms_energy)
            return self._result(
                "NOK", "silence", rms_energy,
                active=self.event_active,
                age=now - self.event_start_time if self.event_active else 0.0,
                locked=self.event_locked_nok,
            )
        self._silence_start_time = None

        if not self.event_active:
            self.event_active = True
            self.event_start_time = now

        event_age = now - self.event_start_time

        if self.event_locked_nok:
            return self._result(
                "NOK", "strong_negative_pattern_locked", rms_energy, age=event_age
            )

        if self.event_confirmed_ok:
            return self._result(
                "OK", "stable_jewel_pattern_confirmed", rms_energy, age=event_age
            )

        is_strong_negative = (
            predicted_class in STRONG_NEGATIVE_CLASSES
            and confidence >= negative_confidence
        )
        if is_strong_negative:
            self.early_negative_hits += 1
            self.consecutive_jewel_hits = 0
            if self.early_negative_hits >= negative_lock_hits:
                self.event_locked_nok = True
                return self._result(
                    "NOK", "strong_negative_pattern_locked", rms_energy, age=event_age
                )
            return self._result(
                "NOK", "strong_negative_pattern_building", rms_energy, age=event_age
            )

        is_verified_jewel = (
            predicted_class == JEWEL_CLASS
            and confidence >= ok_confidence
            and embedding_verified
        )
        if is_verified_jewel:
            self.consecutive_jewel_hits += 1
            self.early_negative_hits = 0
            if self.consecutive_jewel_hits >= required_jewel_hits:
                self.event_confirmed_ok = True
                return self._result(
                    "OK", "stable_jewel_pattern_confirmed", rms_energy, age=event_age
                )
            return self._result(
                "NOK", "verified_jewel_pattern_building", rms_energy, age=event_age
            )

        # Weak nail, background and unverified jewel frames do not decide the event.
        self.early_negative_hits = 0
        self.consecutive_jewel_hits = 0
        reason = (
            "jewel_embedding_rejected"
            if predicted_class == JEWEL_CLASS and confidence >= ok_confidence
            else "waiting_for_clear_pattern"
        )
        return self._result("NOK", reason, rms_energy, age=event_age)

    def _result(
        self,
        final_result: str,
        reason: str,
        rms_energy: float,
        active: bool | None = None,
        age: float | None = None,
        locked: bool | None = None,
    ) -> dict[str, object]:
        return {
            "final_result": final_result,
            "reason": reason,
            "event_active": active if active is not None else self.event_active,
            "event_age": age if age is not None else (
                time.monotonic() - self.event_start_time if self.event_start_time else 0.0
            ),
            "event_locked_nok": locked if locked is not None else self.event_locked_nok,
            "jewel_hits": self.consecutive_jewel_hits,
            "early_negative_hits": self.early_negative_hits,
            "rms_energy": rms_energy,
        }
