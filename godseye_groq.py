#!/usr/bin/env python3
"""
GodsEye v3.0 - GROQ EDITION
Uses Groq API (llama-3.1-8b-instant) - Verified Free Tier
"""

import os
import sys
import json
import sqlite3
import requests
import pandas as pd
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

# ==================== GROQ SETUP (FREE TIER) ====================
print("GodsEye v3.0 - Groq Edition")
print("Free Tier: 30 req/min, 14,400 req/day, $0 cost")
print("=" * 50)

# Get Groq API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("\nGet FREE API key: https://console.groq.com")
    print("Click 'Create API Key' (no credit card required)")
    api_key = input("\nEnter Groq API key (gsk_...): ").strip()
    os.environ["GROQ_API_KEY"] = api_key

GROQ_MODEL = "llama-3.1-8b-instant"  # Fast, good for SQL
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

print(f"✓ Model: {GROQ_MODEL}")
print(f"✓ Free tier limits: 30 req/min, $0 cost\n")

# ==================== CONFIG ====================
DB_FILENAME = "masterwalletsdb.db"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_FILENAME)

# ==================== MAIN CLASS ====================
class GodsEye:
    def __init__(self):
        self.console = Console()
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.schema_cache = {}
        self.session = PromptSession(history=FileHistory(".godseye_history"))
        
        self._load_schema()
        self.console.print(f"[bold green]✓ Database connected: {DB_FILENAME}[/]\n")
    
    def _load_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        self.tables = [t[0] for t in cursor.fetchall()]
        
        for table in self.tables:
            cursor.execute(f"PRAGMA table_info({table})")
            self.schema_cache[table] = [col[1] for col in cursor.fetchall()]
        
        self.schema_text = self._build_schema_text()
        self.console.print(f"[dim]Tables: {', '.join(self.tables)}[/]")
    
    def _build_schema_text(self):
        text = "SQLite Database Schema:\n"
        for table, cols in self.schema_cache.items():
            text += f"- {table}: {', '.join(cols)}\n"
        
        text += """
Rules:
- Cartographers = high win rate wallets
- Convoy = 2+ wallets buying same token within 5 minutes
- Use datetime('now', '-X hours') for time
- Return ONLY JSON: {"sql": "SELECT...", "explanation": "..."}
"""
        return text
    
    def generate_sql(self, question):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": GROQ_MODEL,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": f"You are a SQL expert. Convert questions to SQLite queries.\n{self.schema_text}"
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nRespond with ONLY JSON format:\n{{\"sql\": \"SELECT...\", \"explanation\": \"brief description\"}}"
                }
            ]
        }
        
        try:
            response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 429:
                return None, "Rate limit hit (30 req/min free tier). Wait a few seconds."
            
            response.raise_for_status()
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            content = content.strip()
            if not content.startswith("{"):
                start = content.find("{")
                end = content.rfind("}") + 1
                content = content[start:end] if start != -1 else content
            
            parsed = json.loads(content)
            sql = parsed.get("sql")
            
            # Safety check
            if sql and any(kw in sql.upper() for kw in ["UPDATE ", "DELETE ", "DROP ", "INSERT "]):
                return None, "Blocked destructive query"
            
            return sql, parsed.get("explanation", "Query executed")
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                return None, "Invalid API key. Check your key at console.groq.com"
            return None, f"API error: {e}"
        except Exception as e:
            return None, str(e)
    
    def execute(self, sql):
        if not sql:
            return pd.DataFrame(), 0
        
        start = datetime.now()
        try:
            df = pd.read_sql_query(sql, self.conn)
            elapsed = (datetime.now() - start).total_seconds() * 1000
            return df, elapsed
        except Exception as e:
            self.console.print(f"[red]SQL Error: {e}[/]")
            return pd.DataFrame(), 0
    
    def display(self, df, explanation, elapsed):
        if df.empty:
            self.console.print("[yellow]No data returned.[/]")
            return
        
        self.console.print(f"\n[bold]{explanation}[/] | [dim]{len(df)} rows | {elapsed:.0f}ms[/]")
        
        table = Table(show_header=True, header_style="bold cyan")
        for col in df.columns:
            table.add_column(str(col)[:20])
        
        for _, row in df.head(20).iterrows():
            table.add_row(*[str(x)[:30] for x in row])
        
        self.console.print(table)
        self.console.print()
    
    def run(self):
        self.console.print(Panel.fit("[bold green]GodsEye v3.0 [GROQ MODE][/]\nFree Tier: 30 req/min | Type /exit to quit", border_style="green"))
        
        while True:
            try:
                user_input = self.session.prompt("> ").strip()
                
                if not user_input:
                    continue
                
                # Exit commands
                if user_input.lower() in ["/exit", "/quit", "exit", "quit", "q"]:
                    self.console.print("[red]Shutting down...[/]")
                    self.conn.close()
                    sys.exit(0)
                
                if user_input == "/help":
                    self.console.print("Commands: /tables, /schema, /exit")
                    self.console.print("Ask anything about your wallet data...")
                    continue
                
                if user_input == "/tables":
                    for t, cols in self.schema_cache.items():
                        self.console.print(f"[cyan]{t}[/]: {', '.join(cols[:5])}...")
                    continue
                
                # Query
                sql, explanation = self.generate_sql(user_input)
                
                if sql is None:
                    self.console.print(f"[red]✗ {explanation}[/]")
                    continue
                
                df, elapsed = self.execute(sql)
                self.display(df, explanation, elapsed)
                
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use /exit to quit[/]")
                continue
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/]")

# ==================== RUN ====================
if __name__ == "__main__":
    try:
        GodsEye().run()
    except KeyboardInterrupt:
        print("\nExiting...")