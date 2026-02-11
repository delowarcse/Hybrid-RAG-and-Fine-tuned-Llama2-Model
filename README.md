# Hybrid-RAG-and-Fine-tuned-Llama2-Model
The hybrid LLM solution integrates Retrieval-Augmented Generation (RAG) with a fine-tuned Llama2 model to create an AI researcher assistant specialized in large language models (LLMs), particularly Llama2. The architecture leverages RAG for accurate, context-specific responses from external documents while using the fine-tuned model's internalized knowledge for enhanced reasoning and generalization.
At the core is the RAG pipeline: Documents from the Llama2 research paper (loaded via PyPDFLoader) are split into chunks using CharacterTextSplitter, embedded with HuggingFace's all-MiniLM-L6-v2 model, and stored in a Chroma vector database. Queries trigger similarity search to retrieve relevant chunks (top 5), which are stuffed into a custom prompt template alongside the user's input.

The fine-tuned Llama2 model (accessed via Ollama as "AIresearcher:latest") acts as the generative component. This model, presumably fine-tuned on LLM-related datasets, combines retrieved context with its parametric knowledge. The prompt explicitly instructs the model to prioritize context but fall back on trained knowledge if needed, creating a "hybrid" interaction where RAG mitigates hallucinations and the fine-tuning boosts domain expertise.
The system is wrapped in a Streamlit UI for interactivity, with session state managing conversation history. This architecture addresses limitations of pure RAG (lacks deep reasoning) and fine-tuned models (may forget specifics or hallucinate without grounding).

Rationale for decisions:
- RAG + fine-tuning: Pure RAG with base models can be generic; fine-tuning adds specialization without full retraining. This aligns with course objectives on optimization techniques.
- Chroma for vector store: Lightweight, local, and efficient for small-scale indexing like a single PDF.
- Ollama for model serving: Enables local fine-tuned model use without cloud dependencies, promoting accessibility.
- Custom prompt: Ensures seamless integration, guiding the model to hybridize sources.

System Diagram (ASCII representation):<br>
User Query --> Streamlit UI --> Retrieval Chain <br>
                                   |<br>
                                   v <br>
PDF Loader --> Splitter --> Embeddings --> Chroma DB --> Retriever (Top 5 Chunks)<br>
                                   |<br>
                                   v<br>
Context + Query --> Custom Prompt --> Fine-Tuned Llama2 (Ollama) --> Response<br>

**Implementation Details are available in**<br>
"Hybrid-RAG-and-Fine-tuned-Llama2-Model.pdf"
