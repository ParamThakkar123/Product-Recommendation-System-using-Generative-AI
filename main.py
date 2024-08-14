import streamlit as st
from PIL import Image
import PyPDF2
import io

def main():
    st.title("Hyperpersonalization and Prompt based shopping experience using generative AI")

    st.sidebar.header("Upload files")

    input_type = st.sidebar.selectbox(
        "Choose the type of file you want to upload",
        ("Image", "PDF")
    )

    # Conditional file upload based on selected input type
    if input_type == "Image":
        image_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
        if image_file is not None:
            image = Image.open(image_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

    elif input_type == "PDF":
        pdf_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
        if pdf_file is not None:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            st.write(f"Number of pages in the uploaded PDF: {num_pages}")
            for page in pdf_reader.pages:
                st.text(page.extract_text())

    # Text input at the bottom
    st.write("### Enter a Text Prompt")
    text_prompt = st.text_input("Your Text Prompt")
    if text_prompt:
        st.write("You entered:", text_prompt)

if __name__ == '__main__':
    main()