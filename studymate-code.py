import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import PyPDF2
import io

# Global variables
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None:
        print("Loading IBM Granite 3.2 2B Instruct model...")
        model_name = "ibm-granite/granite-3.2-2b-instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Model loaded successfully!")
    return model, tokenizer

def extract_pdf_text(pdf_files):
    combined_text = ""
    for pdf_file in pdf_files:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                combined_text += page.extract_text() + "\n"
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    return combined_text.strip()

def generate_answer(prompt, max_tokens=512):
    model, tokenizer = load_model()

    formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|assistant|>")[-1].strip()

def qa_function(pdf_files, question):
    if not pdf_files or not question.strip():
        return "Please upload PDF files and ask a question."

    pdf_text = extract_pdf_text(pdf_files)
    if pdf_text.startswith("Error"):
        return pdf_text

    # Limit text length to avoid token limits
    pdf_text = pdf_text[:4000]

    prompt = f"""Based on the following PDF content, answer the question accurately and in detail.

PDF Content:
{pdf_text}

Question: {question}

Answer:"""

    try:
        return generate_answer(prompt, max_tokens=600)
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def summarize_function(pdf_file, summary_type):
    if not pdf_file:
        return "Please upload a PDF file."

    pdf_text = extract_pdf_text([pdf_file])
    if pdf_text.startswith("Error"):
        return pdf_text

    pdf_text = pdf_text[:4500]

    length_instructions = {
        "Brief": "Provide a brief summary in 2-3 sentences highlighting the main points.",
        "Medium": "Provide a medium-length summary in 1-2 paragraphs covering key concepts and details.",
        "Detailed": "Provide a detailed summary in 3-4 paragraphs with comprehensive coverage of all important points."
    }

    prompt = f"""Summarize the following PDF content. {length_instructions[summary_type]}

PDF Content:
{pdf_text}

Summary:"""

    try:
        return generate_answer(prompt, max_tokens=700)
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def extract_keywords_function(pdf_file, num_keywords):
    if not pdf_file:
        return "Please upload a PDF file."

    pdf_text = extract_pdf_text([pdf_file])
    if pdf_text.startswith("Error"):
        return pdf_text

    pdf_text = pdf_text[:4000]

    prompt = f"""Extract the {num_keywords} most important keywords, definitions, and key concepts from the following PDF content.

PDF Content:
{pdf_text}

Please format your response as:

**Keywords:**
1. [Keyword] - [Definition/Explanation]
2. [Keyword] - [Definition/Explanation]

**Key Concepts:**
1. [Concept] - [Detailed explanation]
2. [Concept] - [Detailed explanation]

Extract exactly {num_keywords} items total, focusing on the most important terms and concepts."""

    try:
        return generate_answer(prompt, max_tokens=600)
    except Exception as e:
        return f"Error extracting keywords: {str(e)}"

# Create Gradio Interface
with gr.Blocks(title="AI Document Assistant", theme=gr.themes.Default()) as app:
    gr.Markdown("# 🤖 AI Document Assistant")
    gr.Markdown("Powered by IBM Granite 3.2 2B Instruct")

    with gr.Tabs():
        # Q&A Tab
        with gr.TabItem("❓ Q&A"):
            with gr.Row():
                with gr.Column():
                    qa_pdfs = gr.File(
                        label="Upload PDF(s)",
                        file_count="multiple",
                        file_types=[".pdf"]
                    )
                    qa_question = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask any question about the uploaded PDFs...",
                        lines=3
                    )
                    qa_btn = gr.Button("Get Answer", variant="primary")

                with gr.Column():
                    qa_output = gr.Textbox(
                        label="Answer",
                        lines=12,
                        max_lines=15
                    )

            qa_btn.click(qa_function, [qa_pdfs, qa_question], qa_output)

        # Summarizer Tab
        with gr.TabItem("📝 Summarizer"):
            with gr.Row():
                with gr.Column():
                    sum_pdf = gr.File(
                        label="Upload PDF",
                        file_types=[".pdf"]
                    )
                    sum_type = gr.Radio(
                        choices=["Brief", "Medium", "Detailed"],
                        value="Medium",
                        label="Summary Type"
                    )
                    sum_btn = gr.Button("Generate Summary", variant="primary")

                with gr.Column():
                    sum_output = gr.Textbox(
                        label="Summary",
                        lines=12,
                        max_lines=15
                    )

            sum_btn.click(summarize_function, [sum_pdf, sum_type], sum_output)

        # Keyword Extraction Tab
        with gr.TabItem("🔑 Keywords"):
            with gr.Row():
                with gr.Column():
                    key_pdf = gr.File(
                        label="Upload PDF",
                        file_types=[".pdf"]
                    )
                    key_num = gr.Slider(
                        minimum=5,
                        maximum=15,
                        step=1,
                        value=10,
                        label="Number of Keywords/Concepts"
                    )
                    key_btn = gr.Button("Extract Keywords", variant="primary")

                with gr.Column():
                    key_output = gr.Textbox(
                        label="Keywords & Concepts",
                        lines=12,
                        max_lines=15
                    )

            key_btn.click(extract_keywords_function, [key_pdf, key_num], key_output)

# Launch
if __name__ == "__main__":
    app.launch(share=True, debug=True)
