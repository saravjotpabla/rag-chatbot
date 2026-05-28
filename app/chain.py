from retriever import load_index, retrieve
import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()


def build_prompt():
    prompt = """You are a helpful assistant.Use ONLY the context below to answer the question.If the answer is not in the context, say "I don't know".
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""

    return prompt

def answer_question(query, context, prompt):
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt.format(context=context, question=query)}]
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

    
    
