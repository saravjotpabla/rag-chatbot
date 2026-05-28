from retriever import load_index, retrieve
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()


def build_prompt(history=None):
    if history:
        lines = [
            ("User" if m["role"] == "user" else "Assistant") + ": " + m["content"]
            for m in history
        ]
        history_block = "Conversation so far:\n" + "\n".join(lines) + "\n\n"
    else:
        history_block = ""

    return (
        'You are a helpful assistant. Use ONLY the context below to answer the question. '
        'If the answer is not in the context, say "I don\'t know".\n\n'
        'Context:\n{context}\n\n'
        + history_block
        + 'Current question: {question}\n\nAnswer:'
    )


def answer_question(query, context, prompt):
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt.format(context=context, question=query)}
        ]
    )
    return message

if __name__ == "__main__":
    query = "what are aws best practices?"

    prompt = build_prompt()
    vectorstore, embedding_model = load_index()
    results = retrieve(query, vectorstore, k=3)
    context = ""
    for i, result in enumerate(results):
        context += result.page_content + "\n---\n"

    print("Context for the question:")
    print(context)

    answer = answer_question(query, context, prompt)
    print("Answer from the model:")
    print(answer.content[0].text)

    
    
