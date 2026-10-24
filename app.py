import streamlit as st
import psycopg2
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from indexer import index_pdf

# 1. CONFIGURATION AVEC VOTRE LIEN NEON.TECH
DB_URL = "postgresql://neondb_owner:npg_iMdXy4zUIRP7@ep-winter-rain-b4fuzg82-pooler.c-6.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

st.set_page_config(page_title="RAG Multimodal Agentique", layout="wide")
st.title("🤖 Assistant RAG Multimodal & Agentique")
st.caption("Architecture : LlamaIndex • CLIP • PGVector • LangChain • Qwen (Ollama)")

@st.cache_resource
def get_clip_model():
    return SentenceTransformer('clip-ViT-B-32')

clip_model = get_clip_model()

# 2. DÉFINITION DES 3 OUTILS LANGCHAIN SPÉCIALISÉS (TEXTE, TABLEAUX, IMAGES)
@tool
def chercher_texte(requete: str) -> str:
    """Outil pour rechercher des explications théoriques et des passages de texte dans le PDF."""
    emb = clip_model.encode(str(requete)[:300]).tolist()
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute(
        """SELECT content, page_num FROM documents_multimodaux 
           WHERE doc_type = 'texte'
           ORDER BY embedding <=> %s::vector LIMIT 3;""",
        (str(emb),)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    if not rows:
        return "Aucun texte trouvé."
    return "\n".join([f"--- [TEXTE Page {p}] ---\n{c[:800]}" for c, p in rows])

@tool
def chercher_tableaux(requete: str) -> str:
    """Outil pour rechercher des tableaux de résultats, valeurs numériques et comparaisons dans le PDF."""
    emb = clip_model.encode(str(requete)[:300]).tolist()
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute(
        """SELECT content, page_num FROM documents_multimodaux 
           WHERE doc_type = 'tableau'
           ORDER BY embedding <=> %s::vector LIMIT 2;""",
        (str(emb),)
    )
    rows = cur.fetchall()
    # Si le PDF n'avait pas de tableau détecté par find_tables, on cherche les chiffres dans le texte
    if not rows:
        cur.execute(
            """SELECT content, page_num FROM documents_multimodaux 
               WHERE content ILIKE '%table%' OR content ILIKE '%bleu%' OR doc_type = 'texte'
               ORDER BY embedding <=> %s::vector LIMIT 2;""",
            (str(emb),)
        )
        rows = cur.fetchall()
    cur.close()
    conn.close()
    return "\n".join([f"--- [TABLEAU Page {p}] ---\n{c[:900]}" for c, p in rows])

@tool
def chercher_images(requete: str) -> str:
    """Outil pour rechercher des images, schémas, figures ou diagrammes d'architecture dans le PDF."""
    emb = clip_model.encode(str(requete)[:300]).tolist()
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute(
        """SELECT content, page_num, image_path FROM documents_multimodaux 
           WHERE doc_type = 'image'
           ORDER BY embedding <=> %s::vector LIMIT 2;""",
        (str(emb),)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    if not rows:
        return ""
    return "\n".join([f"IMAGE_TROUVEE|{img_path}|Page {p}|{desc}" for desc, p, img_path in rows])

def routeur_agentique_langchain(question: str):
    """Sélectionne intelligemment et instantanément les outils LangChain selon l'intention."""
    q_low = question.lower()
    outils_choisis = []
    if any(m in q_low for m in ["tableau", "table", "résultat", "valeur", "chiffre", "score", "bleu", "performance"]):
        outils_choisis.append("chercher_tableaux")
    if any(m in q_low for m in ["image", "schéma", "schema", "figure", "architecture", "montre", "diagramme", "visuel"]):
        outils_choisis.append("chercher_images")
    if not outils_choisis or any(m in q_low for m in ["principe", "explique", "comment", "quoi", "attention"]):
        outils_choisis.append("chercher_texte")
    return outils_choisis

# 3. BARRE LATÉRALE : CHARGEMENT D'UN NOUVEAU PDF
with st.sidebar:
    st.header("📄 Document PDF")
    uploaded_file = st.file_uploader("Charger un PDF multimodal", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Indexer le PDF dans PGVector"):
            with open("document.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("Extraction LlamaIndex + Embeddings CLIP en cours..."):
                index_pdf("document.pdf", clip_model=clip_model)
            st.success("PDF indexé avec succès !")

# 4. ORCHESTRATION AGENTIQUE & GÉNÉRATION EN STREAMING (OPTIMISÉE CPU 8 Go RAM)
# num_ctx=2048 et num_predict=350 accélèrent Qwen par 4 sur un CPU AMD Ryzen !
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0, num_ctx=2048, num_predict=350)

question = st.chat_input("Posez votre question sur le texte, les tableaux ou les images du PDF...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        contexte_texte = ""
        tableaux_extraits = []
        images_a_afficher = []

        with st.spinner("Recherche vectorielle dans PGVector..."):
            outils = routeur_agentique_langchain(question)
            
            for nom_outil in outils:
                st.caption(f"🛠️ *Agent LangChain : Appel de l'outil `{nom_outil}`*")
                if nom_outil == "chercher_tableaux":
                    res_tab = chercher_tableaux.invoke({"requete": question})
                    contexte_texte += "\n" + res_tab
                    tableaux_extraits.append(res_tab)
                elif nom_outil == "chercher_texte":
                    contexte_texte += "\n" + chercher_texte.invoke({"requete": question})
                elif nom_outil == "chercher_images":
                    res_img = chercher_images.invoke({"requete": question})
                    for line in res_img.split("\n"):
                        if line.startswith("IMAGE_TROUVEE|"):
                            _, path, page, desc = line.split("|", 3)
                            images_a_afficher.append((path, page, desc))
                            contexte_texte += f"\n[Figure disponible ({page}) : {desc}]\n"

        # Génération en STREAMING : vous voyez les mots s'écrire en temps réel sans jamais bloquer !
        final_prompt = (
            f"Réponds en français de manière claire et concise en utilisant le contexte suivant extrait du PDF.\n"
            f"Cite les numéros de pages sources (ex: [Page X]).\n\n"
            f"CONTEXTE :\n{contexte_texte[:1800]}\n\n"
            f"QUESTION : {question}"
        )
        
        st.write_stream(llm.stream(final_prompt))

        # Affichage propre des tableaux si la question porte sur les tableaux
        if tableaux_extraits:
            st.subheader("📊 Tableaux pertinents extraits du PDF :")
            for tab in tableaux_extraits:
                st.markdown(tab)

        # Affichage des images pertinentes et de leurs descriptions CLIP
        if images_a_afficher:
            st.subheader("🖼️ Figures et Images pertinentes extraites :")
            cols = st.columns(len(images_a_afficher))
            for idx, (img_path, page, desc) in enumerate(images_a_afficher):
                with cols[idx]:
                    st.image(img_path, caption=f"{page} — {desc}", width="stretch")

        with st.expander("🔍 Voir les passages bruts récupérés dans PGVector"):
            st.text(contexte_texte)