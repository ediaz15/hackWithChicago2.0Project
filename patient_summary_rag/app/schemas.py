import pathway as pw

class NoteMetadata(pw.Schema):
    """
    The structured data we want the LLM to extract 
    from every new note during ingestion.
    """
    specialty_tag: str  # e.g., "dermatology", "cardiology", "general"
    record_type: str    # e.g., "lab_result", "doctors_note", "prescription"
    key_findings: str   # A 1-sentence summary of the note's content

class SpecialistQuery(pw.Schema):
    """
    The JSON input your API will expect from the web UI.
    """
    patient_id: str
    specialist_tag: str

class SummaryResponse(pw.Schema):
    """
    The JSON output your API will send back to the web UI.
    """
    summary: str
    sources: list[str] # For citations, as the judges want!