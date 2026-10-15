from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading LLM...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name
)

print("LLM loaded ✅")

def analyze_transcript(transcript):
    prompt = f"""
You are a speech analysis assistant.

Transcript:
{transcript}

Task:
Give a short summary of the transcript.
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )


    return response

def answer_with_context(question, context):
    prompt = f"""
You are a research document assistant.

Answer the question using only the provided context.

If the context does not contain enough information
to answer the question, say:

"I cannot answer this question based on the provided document."

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Give a clear and concise answer.
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return answer


