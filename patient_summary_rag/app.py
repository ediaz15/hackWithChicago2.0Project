import pathway as pw
from pathway.xpacks.llm import document_store as ds
from pathway.xpacks.llm import llms

class PatientSummaryAssistant:
    def __init__(self, document_store, llm):
        self.store = document_store
        self.llm = llm

    @pw.udf
    def generate_summary(self, specialist_type: str, patient_id: str):
        """
        Generate a tailored patient summary for a specific specialist.
        """
        # Retrieve all patient-related documents
        query = f"Patient ID: {patient_id}"
        docs = self.store.retrieve(query=query, k=20)

        # Compose prompt for LLM
        prompt = f"""
        You are a medical assistant generating a patient summary for a {specialist_type}.
        Include only relevant information:
        - Active medications
        - Allergies
        - Key past diagnoses
        - Recent lab results & imaging
        - Recent symptoms
        Cite sources from retrieved documents.
        Do NOT make any diagnosis yourself.
        Documents: {docs}
        """
        return self.llm.chat(prompt)

# Instantiate and serve
document_store = pw.get_component("$document_store")
llm_model = pw.get_component("$llm")
assistant = PatientSummaryAssistant(document_store, llm_model)

server = pw.QASummaryRestServer(
    question_answerer=assistant.generate_summary,
    host="0.0.0.0",
    port=8000
)
server.run()
