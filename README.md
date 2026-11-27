# MaintiDoc — Assistant RAG multimodal pour la maintenance

MaintiDoc permet à un technicien de charger un PDF, de poser une question en français et de vérifier la réponse en consultant les pages, tableaux et figures retrouvés. Ce dépôt correspond à la première étape du projet. Il inclut l’article *Attention Is All You Need* comme document de démonstration.

## Fonctionnement

1. **Extraction** — `indexer.py` utilise PyMuPDF pour lire le texte, détecter les tableaux et extraire les images du PDF. LlamaIndex découpe le texte en passages.
2. **Indexation** — CLIP (`clip-ViT-B-32`) calcule les vecteurs du texte, des tableaux et des images. PostgreSQL avec pgvector conserve chaque vecteur avec son contenu, son type et son numéro de page.
3. **Traitement des images** — Pour chaque image, CLIP sélectionne une catégorie. Le programme la complète avec une légende ou un extrait de la page. La recherche utilise le vecteur visuel et renvoie la description associée.
4. **Recherche** — Dans `app.py`, un routeur par mots-clés sélectionne un ou plusieurs outils LangChain parmi `chercher_texte`, `chercher_tableaux` et `chercher_images`. Par exemple, « score BLEU » déclenche la recherche dans les tableaux.
5. **Réponse** — Qwen 2.5, exécuté localement via Ollama, reçoit la question et le contexte retrouvé. Streamlit affiche la réponse progressivement. L’utilisateur peut ouvrir les passages bruts pour vérifier les citations.

## Installation sous Windows

Prévoir Python 3.11, Git, Ollama et une base PostgreSQL avec l’extension pgvector. Les commandes ci-dessous utilisent PowerShell et Docker pour la base locale. Si vous disposez déjà d’une base compatible, passez directement à la création de la table.

### 1. Télécharger le projet et installer les dépendances

```powershell
git clone https://github.com/AmineZmarrou/RAG_multimodal_agentique.git
cd RAG_multimodal_agentique
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install streamlit psycopg2-binary sentence-transformers pillow pymupdf llama-index-core langchain-core langchain-ollama
```

### 2. Créer la base de démonstration

**Avec Neon**, ouvrez l’éditeur SQL de votre propre base et exécutez directement le bloc SQL ci-dessous. Ce parcours permet de lancer MaintiDoc même si Docker Desktop affiche `exec format error`.

**Avec Docker Desktop**, utilisez le moteur Linux. Dans PowerShell, récupérez sa plateforme et testez le démarrage de l’image pgvector avant de connecter le volume de données.

```powershell
$platform = docker version --format '{{.Server.Os}}/{{.Server.Arch}}'
docker pull --platform $platform pgvector/pgvector:pg16
docker run --rm --platform $platform pgvector/pgvector:pg16 postgres --version
```

Continuez si le test affiche la version de PostgreSQL. Si un ancien conteneur nommé `maintidoc-db` est arrêté avec `exec format error`, conservez-le sous un autre nom avant de recréer le service :

```powershell
docker rename maintidoc-db maintidoc-db-ancien
```

Remplacez `MOT_DE_PASSE_LOCAL` par votre mot de passe. Avec un volume déjà initialisé, utilisez le mot de passe existant.

```powershell
docker run --name maintidoc-db --platform $platform -e POSTGRES_PASSWORD=MOT_DE_PASSE_LOCAL -e POSTGRES_DB=maintidoc -p 127.0.0.1:5432:5432 -v maintidoc-pg:/var/lib/postgresql/data -d pgvector/pgvector:pg16
docker exec maintidoc-db pg_isready -U postgres -d maintidoc
```

Lorsque PostgreSQL affiche `accepting connections`, ouvrez sa console avec `docker exec -it maintidoc-db psql -U postgres -d maintidoc`. Si le test de l’image échoue encore, utilisez le parcours Neon ci-dessus pour poursuivre le projet.

Dans `psql`, exécutez les instructions suivantes. Avec Neon, exécutez-les dans l’éditeur SQL de votre base.

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents_multimodaux (
    id bigserial PRIMARY KEY,
    content text NOT NULL,
    doc_type text NOT NULL,
    page_num integer NOT NULL,
    image_path text,
    embedding vector(512) NOT NULL
);
```

Dans `psql`, saisissez `\q` pour revenir à PowerShell.

### 3. Configurer la connexion

Les trois scripts lisent la variable d’environnement `DATABASE_URL`. Définissez-la dans le même terminal que Streamlit.

Le mot de passe Neon précédemment publié doit être renouvelé par le propriétaire du dépôt. Conservez le nouvel identifiant en dehors du code.

Dans le terminal PowerShell qui servira au lancement, configurez votre base et vérifiez la connexion.

```powershell
$env:DATABASE_URL = "postgresql://postgres:MOT_DE_PASSE_LOCAL@localhost:5432/maintidoc"
.\.venv\Scripts\python.exe test_db.py
```

Avec Neon, utilisez votre propre chaîne de connexion à la place de l’adresse locale. Le script affiche la version de l’extension `vector` si la connexion aboutit.

### 4. Démarrer Ollama et Streamlit

Ouvrez Ollama. Si le service est arrêté, exécutez `ollama serve` dans un deuxième terminal. Dans le terminal du projet, lancez ensuite :

```powershell
ollama pull qwen2.5:1.5b
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Ouvrez l’adresse affichée par Streamlit, généralement <http://localhost:8501>. Le premier chargement de CLIP nécessite une connexion Internet.

## Première démonstration

1. Dans la barre latérale, chargez `attention is all you need.pdf`.
2. Cliquez sur **Indexer le PDF dans PGVector** et attendez la confirmation. Chaque indexation remplace le contenu de la table `documents_multimodaux`. Utilisez une base dédiée à cette démonstration.
3. Posez l’une des questions ci-dessous. Comparez la réponse et la page citée au PDF original.

| Question | Élément à vérifier |
| --- | --- |
| Explique le principe de la self-attention. | Ouvrir les passages bruts et retrouver l’explication sur la page citée. |
| Montre le schéma de l’architecture Transformer. | Comparer la figure affichée et sa légende avec le PDF. |
| Quel score BLEU le tableau indique-t-il pour la traduction anglais-allemand ? | Vérifier le modèle, la colonne et la valeur dans le tableau original. |

Si aucun tableau n’a été détecté, l’outil peut afficher un passage textuel sous l’intitulé « tableau ». Les figures extraites sont enregistrées dans `extracted_images/`. Ce dossier doit rester accessible à l’application.

## Périmètre du prototype

Le prototype traite un PDF à la fois. L’interface accepte les documents PDF et permet de rechercher leurs images intégrées. Le choix des outils repose sur des règles par mots-clés. Chaque question utilise son propre contexte.

Pour démontrer l’usage en maintenance, chargez un manuel contenant une procédure, un schéma de panneau et un tableau de voyants. Le scénario complet prévu par le sujet ajoutera l’analyse d’une photo envoyée par le technicien, la consultation des interventions via MCP et la préparation d’un ticket dont la création exigera une validation humaine. Le fine-tuning, les clients Telegram ou Discord et le déploiement font également partie des étapes suivantes.

MaintiDoc fournit une aide documentaire. Le technicien habilité conserve la responsabilité de l’intervention et l’application ne pilote aucun équipement.