import gradio as gr
from transformers import pipeline

# Load the pre-trained sentiment analysis model
model = pipeline(
    "sentiment-analysis", model="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
)

# Analysis function

def analyze(text):
    result = model(text)
    label = result[0]["label"]
    score = result[0]["score"]
    return f"Sentiment: {label}, Confidence: {score:.2%}"

# Create Gradio interface

demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(lines=5, placeholder="Enter Arabic text here..."),
    outputs="text",
    title="Arabic Sentiment Analysis",
    description="Enter Arabic text to analyze its sentiment using a pre-trained BERT model.",
)

demo.launch()