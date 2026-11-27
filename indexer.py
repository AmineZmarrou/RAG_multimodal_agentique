import os
import fitz  # PyMuPDF
import psycopg2
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

DB_URL = os.environ["DATABASE_URL"]
PDF_PATH = "attention is all you need.pdf"
IMAGE_DIR = "extracted_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

CANDIDATE_LABELS = [
    "diagramme d'architecture réseau ou système",
    "schéma technique de maintenance mécanique",
    "graphique de performance et courbes de résultats",
    "matrice d'attention ou visualisation mathématique",
    "tableau de données numériques",
    "photo d'équipement ou composant"
]

def describe_image_with_clip(clip_model, label_embeddings, image_path, page_text, page_num):
    """Utilise CLIP + le contexte de la page pour générer une description textuelle de l'image."""
    img = Image.open(image_path).convert("RGB")
    img_emb = clip_model.encode(img, convert_to_tensor=True)
    
    cos_scores = util.cos_sim(img_emb, label_embeddings)[0]
    best_label = CANDIDATE_LABELS[int(cos_scores.argmax())]
    
    caption = ""
    for line in page_text.split("\n"):
        if "fig" in line.lower() or "schéma" in line.lower():
            caption += line.strip() + " "
            
    if not caption:
        caption = page_text[:200].replace("\n", " ") + "..."
        
    description = f"Image page {page_num} ({best_label}). Contexte/Légende : {caption}"
    return description, img_emb.tolist()

def index_pdf(pdf_path, clip_model=None):
    if clip_model is None:
        print("Chargement du modèle CLIP (Open Source)...")
        clip_model = SentenceTransformer('clip-ViT-B-32')
        
    label_embeddings = clip_model.encode(CANDIDATE_LABELS, convert_to_tensor=True)
    
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    # Vider la table avant une nouvelle indexation
    cur.execute("TRUNCATE TABLE documents_multimodaux;")
    
    doc = fitz.open(pdf_path)
    splitter = SentenceSplitter(chunk_size=350, chunk_overlap=40)
    
    print(f"Analyse de '{pdf_path}' ({len(doc)} pages)...")
    
    for page_index in range(len(doc)):
        page = doc[page_index]
        page_num = page_index + 1
        page_text = page.get_text("text")
        
        # --- A. EXTRACTION ET INDEXATION DU TEXTE (via LlamaIndex) ---
        if page_text.strip():
            llama_doc = Document(text=page_text, metadata={"page": page_num})
            nodes = splitter.get_nodes_from_documents([llama_doc])
            for node in nodes:
                chunk_text = node.get_content()
                emb = clip_model.encode(chunk_text[:300]).tolist()
                cur.execute(
                    """INSERT INTO documents_multimodaux (content, doc_type, page_num, image_path, embedding)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (chunk_text, "texte", page_num, None, str(emb))
                )
        
        # --- B. EXTRACTION ET INDEXATION DES TABLEAUX ---
        tables = page.find_tables()
        for t_idx, table in enumerate(tables):
            md_table = table.to_markdown()
            if md_table.strip():
                emb = clip_model.encode(md_table[:300]).tolist()
                cur.execute(
                    """INSERT INTO documents_multimodaux (content, doc_type, page_num, image_path, embedding)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (md_table, "tableau", page_num, None, str(emb))
                )
                print(f"  [Page {page_num}] Tableau {t_idx+1} indexé.")

        # --- C. EXTRACTION ET TRAITEMENT DES IMAGES AVEC CLIP ---
        image_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            img_filename = f"{IMAGE_DIR}/page_{page_num}_img_{img_idx+1}.{image_ext}"
            with open(img_filename, "wb") as f:
                f.write(image_bytes)
                
            with Image.open(img_filename) as im:
                if im.width < 50 or im.height < 50:
                    os.remove(img_filename)
                    continue
            
            desc, img_emb = describe_image_with_clip(clip_model, label_embeddings, img_filename, page_text, page_num)
            cur.execute(
                """INSERT INTO documents_multimodaux (content, doc_type, page_num, image_path, embedding)
                   VALUES (%s, %s, %s, %s, %s)""",
                (desc, "image", page_num, img_filename, str(img_emb))
            )
            print(f"  [Page {page_num}] Image {img_idx+1} décrite et indexée avec CLIP.")

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Indexation multimodale terminée avec succès dans PGVector !")

if __name__ == "__main__":
    index_pdf(PDF_PATH)