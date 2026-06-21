import sqlite3
import os
from datetime import datetime
from typing import Dict, Any, Optional, List


class SQLiteDatabase:
    def __init__(self, db_path: str = "data/memory.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_tables()

    def _init_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    occupation TEXT,
                    preferences TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    summary_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    session_id TEXT,
                    keywords TEXT,
                    topics TEXT,
                    summary_text TEXT,
                    created_at TEXT
                )
            ''')

            conn.commit()

    def save_user_profile(self, user_id: str, data: Dict[str, Any]):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO user_profiles 
                (user_id, name, occupation, preferences, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                data.get('name'),
                data.get('occupation'),
                str(data.get('preferences', {})),
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            conn.commit()

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
            row = cursor.fetchone()
            if row:
                return {
                    'user_id': row[0],
                    'name': row[1],
                    'occupation': row[2],
                    'preferences': eval(row[3]),
                    'created_at': row[4],
                    'updated_at': row[5]
                }
            return None

    def save_conversation_summary(self, summary_id: str, user_id: str, session_id: str,
                                   keywords: List[str], topics: List[str], summary_text: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO conversation_summaries
                (summary_id, user_id, session_id, keywords, topics, summary_text, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                summary_id,
                user_id,
                session_id,
                str(keywords),
                str(topics),
                summary_text,
                datetime.now().isoformat()
            ))
            conn.commit()

    def get_user_summaries(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM conversation_summaries 
                WHERE user_id = ? 
                ORDER BY created_at DESC LIMIT ?
            ''', (user_id, limit))
            rows = cursor.fetchall()
            return [{
                'summary_id': row[0],
                'user_id': row[1],
                'session_id': row[2],
                'keywords': eval(row[3]),
                'topics': eval(row[4]),
                'summary_text': row[5],
                'created_at': row[6]
            } for row in rows]
