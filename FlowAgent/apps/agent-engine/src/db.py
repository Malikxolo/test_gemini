"""Database utilities for agent engine."""
import os
import psycopg2
from psycopg2 import extras
from datetime import datetime
from typing import Optional, Dict, Any


def get_db_connection():
    """Get database connection."""
    return psycopg2.connect(os.environ['DATABASE_URL'])


def update_execution_status(
    execution_id: str,
    status: str,
    output: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
    duration_ms: Optional[int] = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    estimated_cost: float = 0.0,
):
    """Update execution status in database."""
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        updates = ['status = %s']
        values = [status]

        if output is not None:
            updates.append('output = %s')
            values.append(extras.Json(output))

        if error is not None:
            updates.append('error = %s')
            values.append(error)

        if started_at is not None:
            updates.append('started_at = %s')
            values.append(started_at)

        if completed_at is not None:
            updates.append('completed_at = %s')
            values.append(completed_at)

        if duration_ms is not None:
            updates.append('duration_ms = %s')
            values.append(duration_ms)

        if input_tokens > 0:
            updates.append('input_tokens = %s')
            values.append(input_tokens)

        if output_tokens > 0:
            updates.append('output_tokens = %s')
            values.append(output_tokens)

        if estimated_cost > 0:
            updates.append('estimated_cost = %s')
            values.append(estimated_cost)

        values.append(execution_id)

        query = f"""
            UPDATE executions
            SET {', '.join(updates)}
            WHERE id = %s
        """

        cur.execute(query, values)
        conn.commit()

    finally:
        cur.close()
        conn.close()
