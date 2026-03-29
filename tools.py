"""
Database Tools - Functions to be used by the Agent.
"""

import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from langchain.tools import Tool

logger = logging.getLogger(__name__)

# --- Connection Settings ---
DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "database": os.getenv("DB_NAME",     "subscription_db"),
    "user":     os.getenv("DB_USER",     "admin"),
    "password": os.getenv("DB_PASSWORD", "sub123"),
    "connect_timeout": 5,
}


def get_db_connection():
    """Establishes a connection to the database and returns the connection object."""
    return psycopg2.connect(**DB_CONFIG)


# ---------------------------------------------------------------------------
# Tool Functions
# ---------------------------------------------------------------------------

def run_sql(query: str) -> str:
    """
    Executes the given SQL query and returns the results in a table format.
    Only SELECT queries are allowed for security reasons.
    """
    query = query.strip()

    # Security: Allow only SELECT statements
    if not query.upper().startswith("SELECT"):
        return "Security Restriction: Only SELECT queries can be executed."

    # Automatically add LIMIT if not present
    if "LIMIT" not in query.upper():
        query = query.rstrip(";") + " LIMIT 100;"

    try:
        conn = get_db_connection()
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                rows = cur.fetchall()
                if not rows:
                    return "Query executed successfully, but returned no results (0 rows)."

                # Get column headers
                cols = list(rows[0].keys())
                header = " | ".join(cols)
                sep    = "-" * len(header)
                lines  = [header, sep]
                for row in rows:
                    lines.append(" | ".join(str(row[c]) for c in cols))

                lines.append(f"\n({len(rows)} rows returned)")
                return "\n".join(lines)
    except psycopg2.Error as e:
        logger.error(f"SQL Error: {e}")
        return f"SQL Error: {e.pgerror or str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return f"Unexpected error: {e}"
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_tables(_=None) -> str:
    """
    Lists all tables in the public schema of the database.
    Use this to see available tables before writing any queries.
    """
    try:
        conn = get_db_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    ORDER BY table_name;
                """)
                tables = [row[0] for row in cur.fetchall()]

        if not tables:
            return "No tables found in the database."
        return "Available tables:\n" + "\n".join(f"  - {t}" for t in tables)
    except Exception as e:
        logger.error(f"get_tables error: {e}")
        return f"Error: {e}"
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_schema(table_name: str) -> str:
    """
    Returns the column names and data types for the specified table.
    Takes only the table name as an argument (e.g., 'subscriptions').
    """
    table_name = table_name.strip().strip("'\"")

    try:
        conn = get_db_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        column_name,
                        data_type,
                        is_nullable,
                        column_default
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name   = %s
                    ORDER BY ordinal_position;
                """, (table_name,))
                rows = cur.fetchall()

        if not rows:
            return f"No table found named '{table_name}'."

        lines = [f"Schema for table '{table_name}':", "-" * 50]
        for col, dtype, nullable, default in rows:
            null_str    = "NULL" if nullable == "YES" else "NOT NULL"
            default_str = f"DEFAULT {default}" if default else ""
            lines.append(f"  {col:30s} {dtype:20s} {null_str} {default_str}".rstrip())

        return "\n".join(lines)
    except Exception as e:
        logger.error(f"get_schema error: {e}")
        return f"Error: {e}"
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_row_count(table_name: str) -> str:
    """
    Returns the total number of records in the specified table.
    """
    table_name = table_name.strip().strip("'\"")
    try:
        conn = get_db_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(f'SELECT COUNT(*) FROM "{table_name}";')  # noqa: S608
                count = cur.fetchone()[0]
        return f"Total records in '{table_name}': {count}"
    except Exception as e:
        return f"Error: {e}"
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# LangChain Tool Objects
# ---------------------------------------------------------------------------

sql_tool = Tool(
    name="run_sql",
    func=run_sql,
    description=(
        "Executes a valid PostgreSQL SELECT query and returns the results. "
        "Input should be a full SQL query (e.g., SELECT * FROM subscriptions WHERE status='active'). "
        "Only SELECT queries are allowed."
    ),
)

tables_tool = Tool(
    name="get_tables",
    func=get_tables,
    description=(
        "Lists all tables in the database. "
        "Call this to see available tables before writing any query. "
        "Requires no arguments."
    ),
)

schema_tool = Tool(
    name="get_schema",
    func=get_schema,
    description=(
        "Returns column names and data types for a specific table. "
        "Takes a table name as an argument (e.g., subscriptions). "
        "Use this to understand the schema before writing SQL."
    ),
)

row_count_tool = Tool(
    name="get_row_count",
    func=get_row_count,
    description=(
        "Returns the total row count of a table. "
        "Takes a table name as an argument (e.g., subscriptions)."
    ),
)