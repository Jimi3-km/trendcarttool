import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# --- Load Model with Feedback and Fallbacks ---
try:
    generator = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.1",
        tokenizer="mistralai/Mistral-7B-Instruct-v0.1",
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
except Exception as e:
    generator = None
    print(f"Error loading model: {e}")

# --- Generate Description Function ---
def generate_description(product_name):
    if not generator:
        return "⚠️ Model failed to load. Please try again later."
    
    if not product_name.strip():
        return "❗ Please enter a product name."

    prompt = (
        f"Write a high-converting Shopify product description for the product: '{product_name}'. "
        "Make it engaging, benefit-driven, and persuasive. Limit to 1–2 powerful sentences."
    )

    try:
        output = generator(prompt, max_length=80, do_sample=True, temperature=0.8)[0]['generated_text']
        return output.strip().split(prompt)[-1].strip()
    except Exception as e:
        return f"❌ An error occurred: {str(e)}"

# --- Gradio UI ---
with gr.Blocks(title="Dropshipping AI Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🛍️ Dropshipping AI Assistant\nGenerate compelling product descriptions for your Shopify store.")

    with gr.Row():
        with gr.Column(scale=2):
            product_input = gr.Textbox(
                label="Product Name", 
                placeholder="e.g., LED Face Mask", 
                lines=1
            )
            generate_btn = gr.Button("✨ Generate Description")
        with gr.Column(scale=3):
            output = gr.Textbox(
                label="AI-Generated Description", 
                lines=3, 
                interactive=False
            )
            copy_btn = gr.Button("📋 Copy to Clipboard")

    status = gr.Textbox(visible=False)

    generate_btn.click(
        fn=generate_description, 
        inputs=product_input, 
        outputs=output,
        api_name="generate_description"
    )

    copy_btn.click(
        fn=lambda text: text, 
        inputs=output, 
        outputs=None,
        js="navigator.clipboard.writeText(arguments[0]); alert('Copied!')"
    )

# --- Launch ---
demo.launch()
