import ollama
from metaflow import FlowSpec, step, IncludeFile, Parameter

class LLMTask:
    def __init__(self, name, prompt_fn):
        self.name = name
        self.prompt_fn = prompt_fn

    def run(self, context):
        prompt = self.prompt_fn(context)
        print(f"\n--- Running Task: {self.name} ---")
        print(f"Prompt:\n{prompt}\n")

        response = ollama.chat(model='llama3.2', messages=[
            {"role": "user", "content": prompt}
        ])
        output = response['message']['content'].strip()
        context[self.name] = output
        print(f"LLM Response:\n{output}\n")
        return output