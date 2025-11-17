# in app/schemas.py
import pathway as pw

class SpecialistQuery(pw.Schema):
    """
    The input query from your web app.
    """
    patient_id: str
    specialist_tag: str

class SummaryResponse(pw.Schema):
    """
    The JSON response your API will send back.
    """
    summary: str
    sources: list[str] # For citations, as the judges want!