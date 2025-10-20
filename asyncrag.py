import sys, os, aiofiles, orjson, asyncio, json, faiss, re, requests, logging
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Callable
from difflib import SequenceMatcher
from pypdf import PdfReader # REAL PDF PARSER

# --- PyQt5 and Async Integration ---
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QTextEdit, QLabel, QFileDialog, QPlainTextEdit, QProgressBar
)
from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt
from qasync import QEventLoop, asyncSlot # For bridging asyncio and PyQt5

# --- Configuration ---
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"
LOG_FILE = "rag_app.log"
CORPUS_FILE = "corpus.json"
MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Logging Setup ---
# Custom handler to redirect logs to a QPlainTextEdit widget
class QTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = QPlainTextEdit(parent)
        self.widget.setReadOnly(True)
        self.widget.setMaximumBlockCount(1000) # Limit log lines
        self.widget.setStyleSheet("font-size: 8pt; color: #404040; background-color: #f0f0f0;")
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        msg = self.format(record)
        # Use a signal to safely update the GUI from any thread
        self.widget.appendPlainText(msg)

# Global logger instance
logger = logging.getLogger('RAG_App')
logger.setLevel(logging.INFO)

# --- 1. REAL PDF PARSER ---

def REAL_PDF_PARSER(pdf_file_path: str, chunk_size=150) -> List[Dict[str, Any]]:
    """
    Real function using pypdf to extract text, chunk it, and format it.
    """
    logger.info(f"Starting real PDF parsing for: {pdf_file_path}")
    docs = []

    try:
        reader = PdfReader(pdf_file_path)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue

            # Basic text cleaning and chunking
            text = re.sub(r'\s+', ' ', text).strip()
            words = text.split()

            # Simple chunking logic
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])

                docs.append({
                    "doc_title": os.path.basename(pdf_file_path),
                    "section_path": f"Page {page_num + 1}",
                    "content": chunk
                })

        logger.info(f"Parsing complete. Created {len(docs)} chunks.")
        return docs

    except Exception as e:
        logger.error(f"Error during PDF parsing: {e}")
        return []

# --- 2. RAG BACKEND (Gemini Integration + Searcher) ---

# Helper function for RAG API call
async def generate_rag_answer(query: str, snippets: List[Dict[str, Any]]) -> str:
    """Calls the Gemini API to generate a grounded answer."""
    logger.info("Calling Gemini API to synthesize answer...")

    if not snippets:
        return "I couldn't find any relevant documents to answer your question."

    context_parts = []
    for i, snippet in enumerate(snippets, 1):
        doc_info = f"Source: {snippet['doc']['doc_title']} > {snippet['doc']['section_path']}"
        context_parts.append(
            f"--- Snippet {i} ---\n"
            f"{doc_info}\n"
            f"Content: {snippet['snippet']}"
        )

    full_context = "\n\n".join(context_parts)

    system_prompt = (
        "You are an expert AI assistant. Answer the user's question STRICTLY "
        "based on the provided context snippets. Do not use external knowledge. "
        "If the context does not contain the answer, state 'The context provided does not contain a specific answer.' "
        "Provide a concise, direct, and well-structured answer, citing the relevant 'Snippet [X]' numbers at the end of each relevant sentence or paragraph."
    )

    user_query = (
        f"Context:\n\n{full_context}\n\n"
        f"--- User Question ---\n"
        f"Based on the context above, answer the following question: '{query}'"
    )

    payload = {
        "contents": [{ "parts": [{ "text": user_query }] }],
        "systemInstruction": {
            "parts": [{ "text": system_prompt }]
        },
        "config": {
            "temperature": 0.1
        }
    }

    # Run synchronous network call in a thread pool (required for standard requests library)
    def sync_fetch():
        try:
            response = requests.post(
                GEMINI_API_URL,
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload),
                timeout=45
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
            return {"error": str(e)}

    # Await the thread execution
    result = await asyncio.to_thread(sync_fetch)

    if "error" in result:
        return f"Error during final answer generation: {result['error']}"

    try:
        text = result['candidates'][0]['content']['parts'][0]['text']
        logger.info("Successfully received LLM response.")
        return text
    except (KeyError, IndexError):
        return "Error: Could not parse response from Gemini API structure."

# Helper functions for Searcher
def encode_sync(model, texts, convert_to_numpy=True, show_progress_bar=False):
    return model.encode(texts, convert_to_numpy=convert_to_numpy, show_progress_bar=show_progress_bar)

def predict_sync(cross, pairs):
    return cross.predict(pairs)

def faiss_use(embeddings: np.ndarray, query_emb: np.ndarray, k: int = 20):
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    query_emb = np.ascontiguousarray(query_emb, dtype=np.float32)

    if embeddings.ndim != 2: raise ValueError("Embeddings must be a 2D array (N, d)")
    N, d = embeddings.shape
    if query_emb.ndim == 1: query_emb = query_emb.reshape(1, -1)

    faiss.normalize_L2(embeddings)
    faiss.normalize_L2(query_emb)

    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    scores, indices = index.search(query_emb, k)
    return scores, indices

# --- RAG Searcher Class (Optimized for Async/Thread Safety) ---

class RAGCore(QObject):
    # Signals to update the UI (must be QObject subclass)
    indexing_progress = pyqtSignal(int)
    status_update = pyqtSignal(str)
    search_complete = pyqtSignal(str, list)

    def __init__(self):
        super().__init__()
        self.corpus: List[Dict[str, Any]] = []
        self.processed: List[str] = []
        self.corpus_embeddings: Optional[np.ndarray] = None

        # Load models on a separate thread (or in main thread before GUI starts)
        self.status_update.emit("Initializing models...")
        self.bi_encoder = SentenceTransformer(MODEL_NAME)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)
        self.status_update.emit("Models loaded.")
        self.executor = ThreadPoolExecutor(max_workers=4)

    @asyncSlot(str)
    async def process_pdf_file(self, file_path: str):
        self.status_update.emit(f"Processing PDF: {os.path.basename(file_path)}...")

        # 1. Parsing (Real-world step)
        loop = asyncio.get_running_loop()
        self.corpus = await loop.run_in_executor(self.executor, REAL_PDF_PARSER, file_path)

        if not self.corpus:
            self.status_update.emit("Error: PDF parsing failed or resulted in no chunks.")
            return

        self.processed = [row["content"] for row in self.corpus]

        # 2. Encoding (The main heavy, blocking task)
        self.status_update.emit(f"Encoding {len(self.corpus)} chunks. This may take a moment...")
        self.indexing_progress.emit(0)

        # Simple progress update for a single encoding run
        # Note: True progress bars require custom iteration in SentenceTransformers or a listener
        # For simplicity, we just show 50% on start and 100% on finish of encoding
        self.indexing_progress.emit(50)

        try:
            self.corpus_embeddings = await loop.run_in_executor(
                self.executor,
                encode_sync,
                self.bi_encoder,
                self.processed,
                True,
                False # Suppress S-T's internal progress bar in non-main thread
            )
            self.indexing_progress.emit(100)
            self.status_update.emit(f"Corpus ready! {len(self.corpus)} chunks indexed.")
            logger.info("Corpus indexing complete.")

        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            self.status_update.emit(f"Error: Encoding failed. {e}")


    @asyncSlot(str, int)
    async def perform_rag_search(self, query: str, rerank_top: int):
        self.status_update.emit(f"Searching for: '{query[:30]}...'")

        if self.corpus_embeddings is None:
            self.status_update.emit("Error: Corpus not indexed. Please process a PDF first.")
            return

        loop = asyncio.get_running_loop()

        # 1. Encode Query
        query_emb_list = await loop.run_in_executor(self.executor, encode_sync, self.bi_encoder, [query], True, False)
        query_emb = query_emb_list[0]

        # 2. Bi-Encoder (FAISS) Search - top 20
        scores_faiss, indices_faiss = await loop.run_in_executor(
            self.executor,
            faiss_use,
            self.corpus_embeddings,
            query_emb,
            20 # top_k
        )

        candidates = []
        for cid, score in zip(indices_faiss[0], scores_faiss[0]):
            candidates.append((self.processed[cid], self.corpus[cid], int(cid), float(score)))

        # 3. Cross-Encoder (Reranking)
        pairs = [(query, cand[0]) for cand in candidates]

        if not pairs:
            self.status_update.emit("No relevant snippets found.")
            self.search_complete.emit("No relevant snippets found.", [])
            return

        scores = await loop.run_in_executor(self.executor, predict_sync, self.cross_encoder, pairs)

        reranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        top_snippets = []

        for rank, (score, (flat_text, raw_doc, orig_idx, embed_score)) in enumerate(reranked[:rerank_top], start=1):
            top_snippets.append({
                "rank": rank,
                "rerank_score": float(score),
                "embed_score": float(embed_score),
                "doc_index": int(orig_idx),
                # Clean snippet for LLM context
                "snippet": re.sub(r'\s+', ' ', flat_text).strip(),
                "doc": raw_doc
            })

        logger.info(f"Reranking complete. Top {rerank_top} snippets selected.")

        # 4. LLM Synthesis
        final_answer = await generate_rag_answer(query, top_snippets)

        self.status_update.emit("Final answer generated successfully.")
        self.search_complete.emit(final_answer, top_snippets)

# --- 3. PYQT5 UI ---

class RAGWindow(QMainWindow):
    def __init__(self, rag_core):
        super().__init__()
        self.rag_core = rag_core
        self.setWindowTitle("Gemini RAG App (PyQt5 + Async)")
        self.setGeometry(100, 100, 1000, 700)

        self.init_logging()
        self.init_ui()
        self.connect_signals()

        # Initial status check
        if API_KEY == "YOUR_API_KEY_HERE":
            logger.warning("GEMINI_API_KEY is not set! Set the environment variable or replace the placeholder in code.")

    def init_logging(self):
        # File handler
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

        # UI log handler
        self.log_widget_handler = QTextEditLogger(self)
        self.log_widget_handler.setLevel(logging.INFO)
        logger.addHandler(self.log_widget_handler)

        logger.info("Application starting up.")

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Top Controls (File Select) ---
        file_layout = QHBoxLayout()
        self.file_path_label = QLabel("No PDF selected")
        self.file_path_label.setStyleSheet("font-weight: bold;")
        self.select_file_btn = QPushButton("Select PDF")
        self.process_pdf_btn = QPushButton("Index PDF")
        self.process_pdf_btn.setEnabled(False)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        file_layout.addWidget(self.file_path_label)
        file_layout.addWidget(self.select_file_btn)
        file_layout.addWidget(self.process_pdf_btn)
        file_layout.addWidget(self.progress_bar)
        main_layout.addLayout(file_layout)

        # --- Query and Search ---
        query_layout = QHBoxLayout()
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter your question here (e.g., 'What is the role of max_features in Random Forest?')")
        self.search_btn = QPushButton("Search RAG")
        self.search_btn.setEnabled(False)
        self.rerank_spin = QLineEdit("3")
        self.rerank_spin.setFixedWidth(50)

        query_layout.addWidget(QLabel("Query:"))
        query_layout.addWidget(self.query_input)
        query_layout.addWidget(QLabel("Top K:"))
        query_layout.addWidget(self.rerank_spin)
        query_layout.addWidget(self.search_btn)
        main_layout.addLayout(query_layout)

        # --- Output Area ---
        output_layout = QHBoxLayout()

        # Answer/Summary
        answer_box = QVBoxLayout()
        answer_box.addWidget(QLabel("LLM Grounded Answer:"))
        self.answer_text = QTextEdit()
        self.answer_text.setReadOnly(True)
        answer_box.addWidget(self.answer_text)

        # Context Snippets (Reranked)
        context_box = QVBoxLayout()
        context_box.addWidget(QLabel("Top Context Snippets:"))
        self.context_text = QTextEdit()
        self.context_text.setReadOnly(True)
        context_box.addWidget(self.context_text)

        output_layout.addLayout(answer_box)
        output_layout.addLayout(context_box)
        main_layout.addLayout(output_layout)

        # --- Status and Logs ---
        main_layout.addWidget(QLabel("Logs / Status:"))
        main_layout.addWidget(self.log_widget_handler.widget)

    def connect_signals(self):
        # UI Button Actions
        self.select_file_btn.clicked.connect(self.select_pdf)
        self.process_pdf_btn.clicked.connect(self.process_pdf)
        self.search_btn.clicked.connect(self.search_rag)

        # RAG Core Signals
        self.rag_core.status_update.connect(self.update_status)
        self.rag_core.indexing_progress.connect(self.update_progress)
        self.rag_core.search_complete.connect(self.display_results)

    def select_pdf(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select PDF File", "", "PDF Files (*.pdf)")
        if file_name:
            self.file_path_label.setText(file_name)
            self.current_pdf_path = file_name
            self.process_pdf_btn.setEnabled(True)
            self.search_btn.setEnabled(False)
            self.answer_text.clear()
            self.context_text.clear()
            logger.info(f"PDF selected: {file_name}")

    @asyncSlot()
    async def process_pdf(self):
        self.process_pdf_btn.setEnabled(False)
        self.select_file_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.search_btn.setEnabled(False)
        self.answer_text.clear()
        self.context_text.clear()

        # Start the async process in the RAGCore object
        await self.rag_core.process_pdf_file(self.current_pdf_path)

        # Re-enable buttons once indexing is complete (indicated by status_update)
        if self.rag_core.corpus_embeddings is not None:
             self.process_pdf_btn.setEnabled(True)
             self.select_file_btn.setEnabled(True)
             self.search_btn.setEnabled(True)
        else:
            # If indexing failed, re-enable only select button
            self.select_file_btn.setEnabled(True)

    @asyncSlot()
    async def search_rag(self):
        query = self.query_input.text().strip()
        rerank_top = int(self.rerank_spin.text()) if self.rerank_spin.text().isdigit() else 3

        if not query:
            logger.warning("Query input is empty.")
            return

        self.search_btn.setEnabled(False)
        self.answer_text.setText("Searching and synthesizing answer... please wait.")
        self.context_text.clear()

        # Start the async RAG search
        await self.rag_core.perform_rag_search(query, rerank_top)

        self.search_btn.setEnabled(True)

    def update_status(self, message: str):
        logger.info(f"STATUS: {message}")

    def update_progress(self, value: int):
        self.progress_bar.setValue(value)
        if value == 100:
            self.progress_bar.setVisible(False)

    def display_results(self, answer: str, snippets: List[Dict[str, Any]]):
        self.answer_text.setText(answer)

        context_display = ""
        for snippet in snippets:
            doc = snippet['doc']
            context_display += (
                f"--- Snippet {snippet['rank']} (Score: {snippet['rerank_score']:.4f}) ---\n"
                f"Source: {doc['doc_title']} > {doc['section_path']}\n"
                f"{snippet['snippet']}\n\n"
            )

        self.context_text.setText(context_display)

        logger.info("Search results displayed in UI.")

# --- 4. Main Execution ---

if __name__ == "__main__":
    # Ensure all blocking initialization happens before the GUI event loop starts
    try:
        # 1. Initialize RAG Core (Models load here)
        rag_core = RAGCore()
    except Exception as e:
        print(f"FATAL ERROR during RAG core initialization: {e}")
        sys.exit(1)

    # 2. Start PyQt5 Application
    app = QApplication(sys.argv)

    # 3. Integrate PyQt5 and asyncio event loops
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    # 4. Create and Show Window
    window = RAGWindow(rag_core)
    window.show()

    # 5. Execute the application with the QEventLoop
    with loop:
        sys.exit(loop.run_forever())
