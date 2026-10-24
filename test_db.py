import psycopg2

# Remplacez par VOTRE lien copié à l'Étape 2
DB_URL = "postgresql://neondb_owner:npg_iMdXy4zUIRP7@ep-winter-rain-b4fuzg82-pooler.c-6.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

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