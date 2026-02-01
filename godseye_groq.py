#!/usr/bin/env python3
"""
GodsEye v3.0 - GROQ EDITION
Uses Groq API (llama-3.1-8b-instant) - Verified Free Tier
"""

import os
import sys
import sqlite3
import re
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
        text = "SCHEMA:\n"
        for table, cols in self.schema_cache.items():
            text += f"- {table}: {', '.join(cols)}\n"

        text += """
SQLITE DIALECT RULES:
- Use only SQLite syntax and functions (no DATEDIFF).
- Use strftime('%s', ...) or datetime('now', '-X hours') for time comparisons.
- Use only tables/columns listed in SCHEMA.
- Return exactly one SQL statement inside a fenced code block:
```sql
SELECT ...
```
"""
        return text.strip()

    def _build_prompt(self, question, extra_instructions=None):
        extra = f"\nExtra instructions: {extra_instructions}\n" if extra_instructions else ""
        return (
            "You are a SQL expert. Convert questions to SQLite queries.\n\n"
            f"{self.schema_text}\n\n"
            f"Question: {question}\n"
            f"{extra}"
            "Return only the SQL in a single fenced code block as specified."
        )

    def _extract_sql(self, content):
        if "```sql" not in content:
            return None
        try:
            return content.split("```sql", 1)[1].split("```", 1)[0].strip()
        except IndexError:
            return None

    def _extract_tables(self, sql):
        tables = set()
        aliases = {}
        for match in re.finditer(
            r"\b(from|join)\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+(as\s+)?([A-Za-z_][A-Za-z0-9_]*))?",
            sql,
            re.IGNORECASE,
        ):
            table = match.group(2)
            alias = match.group(4)
            tables.add(table)
            if alias:
                aliases[alias] = table
        return tables, aliases

    def _extract_columns(self, sql, aliases):
        columns = set()
        for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b", sql):
            table_or_alias, column = match.groups()
            table = aliases.get(table_or_alias, table_or_alias)
            columns.add((table, column))
        return columns

    def _validate_sql(self, sql):
        if not sql:
            return False, "Empty SQL."
        if any(kw in sql.upper() for kw in ["UPDATE ", "DELETE ", "DROP ", "INSERT ", "ALTER "]):
            return False, "Blocked destructive query."

        tables, aliases = self._extract_tables(sql)
        unknown_tables = [t for t in tables if t not in self.schema_cache]
        if unknown_tables:
            return False, f"Unknown tables used: {', '.join(sorted(unknown_tables))}."

        columns = self._extract_columns(sql, aliases)
        unknown_columns = []
        for table, column in columns:
            if table not in self.schema_cache or column not in self.schema_cache.get(table, []):
                unknown_columns.append(f"{table}.{column}")

        if unknown_columns:
            return False, f"Unknown columns used: {', '.join(sorted(set(unknown_columns)))}."

        bare_columns = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", sql))
        alias_names = set(aliases.keys())
        alias_names.update(re.findall(r"\bas\s+([A-Za-z_][A-Za-z0-9_]*)\b", sql, re.IGNORECASE))
        sql_keywords = {
            "select", "from", "where", "join", "left", "right", "inner", "outer", "on",
            "group", "by", "order", "limit", "as", "and", "or", "not", "null", "is",
            "in", "like", "distinct", "case", "when", "then", "else", "end", "asc", "desc",
            "count", "sum", "avg", "min", "max", "abs", "coalesce", "strftime", "datetime",
            "date", "time", "cast", "having"
        }
        table_names = set(self.schema_cache.keys())
        known_columns = {col for cols in self.schema_cache.values() for col in cols}
        for token in bare_columns:
            token_lower = token.lower()
            if token_lower in sql_keywords:
                continue
            if token in table_names:
                continue
            if token in alias_names:
                continue
            if token.isdigit():
                continue
            if token not in known_columns:
                return False, f"Unknown column used: {token}."

        return True, "SQL validated."

    def generate_sql(self, question, extra_instructions=None):
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
                    "content": self._build_prompt(question, extra_instructions)
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
            sql = self._extract_sql(content)

            return sql, "Query executed"
            
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

    def _match_template(self, user_input):
        normalized = user_input.lower()
        if "top wallets by volume" in normalized:
            sql = (
                "SELECT scan_wallet, SUM(ABS(balance_delta_sol)) AS total_volume_sol "
                "FROM tx WHERE err IS NULL "
                "GROUP BY scan_wallet "
                "ORDER BY total_volume_sol DESC "
                "LIMIT 10"
            )
            return sql, "Top wallets by volume"
        if "success rate" in normalized:
            sql = (
                "SELECT scan_wallet, "
                "ROUND(100.0 * SUM(CASE WHEN err IS NULL THEN 1 ELSE 0 END) / COUNT(*), 2) AS success_rate_pct, "
                "COUNT(*) AS total_txs "
                "FROM tx "
                "GROUP BY scan_wallet "
                "ORDER BY success_rate_pct DESC"
            )
            return sql, "Success rate by wallet"
        if "most profitable wallets + names" in normalized or "most profitable wallets and names" in normalized:
            sql = (
                "SELECT w.wallet_label, t.scan_wallet, SUM(t.balance_delta_sol) AS total_profit_sol "
                "FROM tx t "
                "JOIN wallets w ON w.wallet_address = t.scan_wallet "
                "WHERE t.err IS NULL "
                "GROUP BY w.wallet_label, t.scan_wallet "
                "ORDER BY total_profit_sol DESC "
                "LIMIT 10"
            )
            return sql, "Most profitable wallets with names"
        if "most profitable wallets" in normalized:
            sql = (
                "SELECT scan_wallet, SUM(balance_delta_sol) AS total_profit_sol "
                "FROM tx WHERE err IS NULL "
                "GROUP BY scan_wallet "
                "ORDER BY total_profit_sol DESC "
                "LIMIT 10"
            )
            return sql, "Most profitable wallets"
        return None, None

    def _regenerate_sql(self, question, reason, attempts=2):
        for _ in range(attempts):
            extra = f"The previous SQL was invalid. Reason: {reason}. Use only valid tables/columns."
            sql, explanation = self.generate_sql(question, extra_instructions=extra)
            if sql:
                valid, validation_message = self._validate_sql(sql)
                if valid:
                    return sql, explanation
                reason = validation_message
        return None, f"Failed to generate valid SQL after {attempts} attempts. Last error: {reason}"

    def _fallback_sql(self, question):
        extra = (
            "The previous SQL returned zero rows. Relax constraints by removing time filters, "
            "simplifying joins, or removing strict filters. Return a broader query."
        )
        sql, explanation = self.generate_sql(question, extra_instructions=extra)
        if sql:
            valid, validation_message = self._validate_sql(sql)
            if valid:
                return sql, explanation
        return None, "Fallback SQL generation failed."
    
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
                
                template_sql, template_explanation = self._match_template(user_input)
                if template_sql:
                    sql, explanation = template_sql, template_explanation
                else:
                    sql, explanation = self.generate_sql(user_input)
                
                if sql is None:
                    if not template_sql:
                        sql, explanation = self._regenerate_sql(user_input, "Missing SQL code block.")
                        if sql is None:
                            self.console.print(f"[red]✗ {explanation}[/]")
                            continue
                    else:
                        self.console.print(f"[red]✗ {explanation}[/]")
                        continue

                valid, validation_message = self._validate_sql(sql)
                if not valid:
                    sql, explanation = self._regenerate_sql(user_input, validation_message)
                    if sql is None:
                        self.console.print(f"[red]✗ {explanation}[/]")
                        continue
                
                df, elapsed = self.execute(sql)
                if df.empty and not template_sql:
                    fallback_sql, fallback_explanation = self._fallback_sql(user_input)
                    if fallback_sql:
                        df, elapsed = self.execute(fallback_sql)
                        if not df.empty:
                            self.display(df, fallback_explanation, elapsed)
                            continue
                    self.console.print("[yellow]No data returned.[/]")
                    self.console.print(f"[dim]Executed SQL: {sql}[/]")
                    self.console.print("[dim]Reason guess: filters may be too strict or joins too limiting.[/]")
                    continue

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
