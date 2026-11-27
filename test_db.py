import os
import psycopg2

DB_URL = os.environ["DATABASE_URL"]

try:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    # Vérifier que l'extension vector est bien active
    cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
    res = cur.fetchone()
    print(f"Connexion réussie ! Extension installée : {res[0]} (version {res[1]})")
    
    cur.close()
    conn.close()
except Exception as e:
    print("Erreur de connexion :", e)