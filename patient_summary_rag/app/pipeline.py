import os
import pathway as pw
from dotenv import load_dotenv
from pathway.xpack.llm.embedders import OpenAIEmbedder
from pathway.xpack.llm.models import OpenAIModel
from pathway.xpack.llm.question_answering import RAGQuestionAnswerer
from pathway.xpack.llm.splitters import TokenCountSplitter
from pathway.xpack.llm.externals import extract

# Import your custom data schemas
from app.schemas import NoteMetadata, SpecialistQuery, SummaryResponse

# --- 1. LOAD CONFIGURATION ---
# Load variables from your .env file (DATABASE_URL, OPENAI_API_KEY, etc.)
load_dotenv()
DB_URL = os.environ.get("DATABASE_URL")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# (Placeholder for your Aparavi API key)
# APARAVI_API_KEY = os.environ.get("APARAVI_API_KEY") 

# --- (PLACEHOLDER) APARAVI TOOLCHAIN ---
# This is a mock function. You will replace this with the
# actual Aparavi SDK call to redact PII/PHI.
@pw.udf
def redact_patient_note(note_text: str) -> str:
    """
    This function represents your Aparavi security station.
    It takes raw text and returns secured, redacted text.
    """
    # In production, you would call:
    # return aparavi.redact(note_text, policies=["phi", "pii"])
    
    # For the hackathon, you can start with a simple placeholder
    # to prove the pipeline works, then integrate Aparavi.
    print("Redacting text (Aparavi)...")
    return note_text.replace("John Doe", "[REDACTED_NAME]")

# --- 2. DEFINE MODELS AND PROMPTS ---

# Initialize your OpenAI models
embedder = OpenAIEmbedder(
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)

llm = OpenAIModel(
    api_key=OPENAI_API_KEY,
    model="gpt-4o"
)

# Define the prompt for your Query API
prompt_template = """
You are a medical AI assistant. Your task is to generate a 1 to 2 page summary
for a {specialist_tag} specialist regarding Patient {patient_id}.
Use ONLY the following context to write your summary.
You MUST include citations for each fact using the 'source_doc' metadata.

CONTEXT:
{context}

QUERY:
Generate a full, detailed summary.
"""

# --- 3. THE "LIVE INGESTION" PIPELINE ---

print("Starting ingestion pipeline...")

# Connect to your Supabase Postgres DB using the CDC connector
# This listens for any INSERTs into the 'notes' table
live_notes_stream = pw.io.postgres.cdc(
    dsn=DB_URL,
    table="notes",
    publication_name="pathway_pub"  # The channel you created
)

# Add a 'source_doc' column for citations
notes_with_source = live_notes_stream.with_columns(
    source_doc=pw.apply(
        lambda created_at, id: f"Note_{created_at.date()}_{id.split('-')[0]}.txt",
        pw.this.created_at, pw.this.id
    )
)

# Apply the Aparavi redaction
safe_notes_stream = notes_with_source.with_columns(
    safe_text=redact_patient_note(pw.this.note_text)
)

# Apply the LLM "Smart Tagger" to extract metadata
tagged_notes_stream = safe_notes_stream.with_columns(
    metadata=extract(
        pw.this.safe_text,
        schema=NoteMetadata,
        llm=llm
    )
)

# Unpack the metadata into top-level columns for filtering
tagged_notes_stream = tagged_notes_stream.with_columns(
    specialty_tag=pw.this.metadata.specialty_tag,
    record_type=pw.this.metadata.record_type
)

# Chunk the safe text for embedding
chunks_stream = TokenCountSplitter(
    data=tagged_notes_stream,
    text_column="safe_text",
    max_tokens=512
)

# Embed the chunks
embedded_chunks = embedder(
    data=chunks_stream.chunks,
    text_column="text"
)

# Create the Live Vector Store (Index)
live_index = pw.vector_store.index(
    embedded_chunks,
    embedder=embedder,
    vector_column="embedding",
    metadata_columns=[
        "patient_id",
        "specialty_tag",
        "record_type",
        "source_doc"
    ]
)

print("Ingestion pipeline is live and listening.")

# --- 4. THE "ON-DEMAND QUERY" PIPELINE ---

@pw.udf
def build_specialist_rag_app(
    query: SpecialistQuery,
    index: pw.Table,
) -> SummaryResponse:
    
    # 1. Filter the LIVE index by patient_id
    patient_index = index.filter(
        pw.this.patient_id == query.patient_id
    )
    
    # 2. Filter AGAIN by the requested specialist_tag
    specialist_index = patient_index.filter(
        pw.this.specialty_tag == query.specialist_tag
    )

    # 3. Create a RAG app on-the-fly using ONLY the filtered data
    rag_app = RAGQuestionAnswerer(
        index=specialist_index,
        embedder=embedder,
        llm=llm,
        prompt_template=prompt_template.format(
            specialist_tag=query.specialist_tag, 
            patient_id=query.patient_id
        )
    )

    # 4. Run the RAG query
    response = rag_app(
        query="Generate a full summary.",
        metadata_to_include=["source_doc"] # For citations
    )

    # 5. Format and return the JSON response
    sources = [meta.get("source_doc", "Unknown Source") for meta in response.metadata]
    
    return SummaryResponse(
        summary=response.response,
        sources=list(set(sources)) # Return a unique list of sources
    )

# Wire the API: Connect the UDF to an HTTP endpoint
rag_service = pw.udf(build_specialist_rag_app)(
    query=pw.io.http.read_body(SpecialistQuery),
    index=live_index
)

# --- 5. RUN THE APPLICATION ---

# Serve your RAG service on port 8080 at the /summarize endpoint
pw.io.http.rest_server(
    port=8080,
    endpoints={"/summarize": rag_service},
    with_basic_ui=True  # Provides a simple test UI at http://localhost:8080
)

# This command starts both the ingestion pipeline (listening to Supabase)
# and the query API server.
print("Starting Pathway application server on http://localhost:8080")
pw.run()