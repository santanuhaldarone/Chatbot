import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

def load_model():
    model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

def get_answer(context, question, tokenizer, model):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    with torch.no_grad():
        outputs = model(**inputs)
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1

    answer_ids = input_ids[start:end]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
    return answer

st.title("Simple QA Chatbot")

st.write("Enter some context and ask a question. The bot will try to answer based on the context.")

context = st.text_area("Enter context")
question = st.text_input("Enter your question")

if st.button("Get Answer"):
    if context and question:
        tokenizer, model = load_model()
        answer = get_answer(context, question, tokenizer, model)
        st.write("**Answer:**", answer)
    else:
        st.write("Please enter both context and question.")

st.markdown("---")
st.caption("Made with ❤️ by Santanu")
