"""
Event and Detection Persistence Module (SQLite + CSV).
"""

import os
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from backend.config import settings, OUTPUTS_DIR
from backend.schemas import EventItem

class StorageManager:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.db_path
        self.detections_csv = OUTPUTS_DIR / "detections.csv"
        self.events_csv = OUTPUTS_DIR / "events.csv"
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    video_timestamp REAL,
                    video_timestamp_str TEXT,
                    processing_timestamp TEXT,
                    object_class TEXT,
                    raw_class TEXT,
                    tracking_id INTEGER,
                    confidence REAL,
                    zone_id TEXT,
                    zone_name TEXT,
                    event_type TEXT,
                    barrier_state TEXT,
                    risk_score REAL,
                    severity TEXT,
                    evidence_path TEXT,
                    notes TEXT
                )
            """)
            conn.commit()

    def save_event(self, event: EventItem):
        """Save a single event into SQLite and append to events.csv."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO events (
                    event_id, video_timestamp, video_timestamp_str, processing_timestamp,
                    object_class, raw_class, tracking_id, confidence, zone_id, zone_name,
                    event_type, barrier_state, risk_score, severity, evidence_path, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id, event.video_timestamp, event.video_timestamp_str,
                event.processing_timestamp, event.object_class, event.raw_class,
                event.tracking_id, event.confidence, event.zone_id, event.zone_name,
                event.event_type, event.barrier_state, event.risk_score, event.severity,
                event.evidence_path, event.notes
            ))
            conn.commit()

        # Update events CSV
        self._append_event_to_csv(event)

    def update_event_evidence(self, event_id: str, evidence_path: str):
        """Update the evidence path for an existing event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE events SET evidence_path = ? WHERE event_id = ?", (evidence_path, event_id))
            conn.commit()

    def get_events(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        object_class: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query stored events with optional filtering."""
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        if object_class:
            query += " AND object_class = ?"
            params.append(object_class)
        query += " ORDER BY video_timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_event_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events WHERE event_id = ?", (event_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def record_detections_batch(self, detections_batch: List[Dict[str, Any]]):
        """Append a batch of frame detections to outputs/detections.csv."""
        if not detections_batch:
            return
        df = pd.DataFrame(detections_batch)
        write_header = not os.path.exists(self.detections_csv)
        df.to_csv(self.detections_csv, mode="a", header=write_header, index=False)

    def _append_event_to_csv(self, event: EventItem):
        df = pd.DataFrame([event.model_dump()])
        write_header = not os.path.exists(self.events_csv)
        df.to_csv(self.events_csv, mode="a", header=write_header, index=False)

    def clear_session_data(self):
        """Clear database and CSVs for a clean test run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM events")
            conn.commit()
        if os.path.exists(self.detections_csv):
            os.remove(self.detections_csv)
        if os.path.exists(self.events_csv):
            os.remove(self.events_csv)

storage_manager = StorageManager()
