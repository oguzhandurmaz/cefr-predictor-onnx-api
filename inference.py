import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import gradio as gr
import numpy as np

# 1. Settings and RoadMap
# 'cefr_deberta_lora' model path
model_path = "models/cefr_deberta_lora" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# 2. Explainability Function (Calculates Word Importance)
def get_word_importance(text):
    labels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    words = text.split()
    if not words:
        return "", {}

    # Original Prediction
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        base_probs = F.softmax(model(**inputs).logits, dim=-1).cpu().numpy()[0]
    
    base_idx = np.argmax(base_probs)
    base_label = labels[base_idx]
    
    # Word by word importance analysis (Simple LIME logic)
    importance_scores = []
    for i in range(len(words)):
        # Remove the current word and predict again
        temp_words = words[:i] + words[i+1:]
        temp_text = " ".join(temp_words)
        
        if not temp_text.strip():
            importance_scores.append((words[i], 0))
            continue
            
        temp_inputs = tokenizer(temp_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            temp_probs = F.softmax(model(**temp_inputs).logits, dim=-1).cpu().numpy()[0]
        
        # How much did the absence of this word reduce the prediction?
        score = base_probs[base_idx] - temp_probs[base_idx]
        importance_scores.append((words[i], score))

    # Convert to Gradio HighlightedText format
    # Normalize scores to -1 to +1
    max_s = max([abs(s) for w, s in importance_scores]) if importance_scores else 1
    highlighted_output = []
    for word, score in importance_scores:
        normalized_score = score / max_s if max_s != 0 else 0
        highlighted_output.append((word + " ", normalized_score))

    confidences = {labels[i]: float(base_probs[i]) for i in range(len(labels))}
    return highlighted_output, confidences

# 3. Gradio Interface (Blocks Structure for Best Performance)
with gr.Blocks() as demo:
    gr.Markdown("# 📖 CEFR Level Analysis and Word Importance Visualizer")
    gr.Markdown("Enter the text; the model will predict the level and show which word has how much effect on this decision with colors.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(lines=5, label="English Text", placeholder="Enter text here...")
            btn = gr.Button("Analyze", variant="primary")
        
        with gr.Column():
            output_label = gr.Label(num_top_classes=3, label="Level Prediction")
            # Word importance component
            output_highlight = gr.HighlightedText(
                label="Word Importance Analysis (Bold = High Importance)",
                combine_adjacent=False,
                show_legend=True,
                color_map={"+": "red", "-": "blue"} # Positive effects red, negative effects blue
            )

    btn.click(get_word_importance, inputs=input_text, outputs=[output_highlight, output_label])
    
    gr.Examples(
        examples=[["This is a very sophisticated and ubiquitous example."], ["Hello, I am a student."]],
        inputs=input_text
    )


if __name__ == "__main__":
    demo.launch(share=True)