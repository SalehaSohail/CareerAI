"""It is necessary to make sure that any code related to the resumés
does not get mixed up with the database of the jobs and any other
matching algorithms. This stage will involve CareerAI working directly
with the text of the resumés so that we can create and test our
matching pipeline first and then develop the module further and allow
it to process PDF or DOCX resumés and extract text information from
them.

Having resume processing in a separate module makes it easier to
work with the project and to implement different formats of resumés
in the future."""


from pypdf import PdfReader


def extract_text_from_pdf(file_path):
    """
    Extract text from all pages of a PDF resume.
    """

    reader = PdfReader(file_path)

    extracted_text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            extracted_text += page_text + "\n"

    return extracted_text