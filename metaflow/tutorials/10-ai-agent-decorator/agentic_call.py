import ollama
from metaflow import user_step_decorator, current
from llmtask import LLMTask

@user_step_decorator
def agentic_call(step_name, flow, inputs=None, attributes=None):
    yield
    context = {}
    # Step 1: Run task1. This task can be read from config files, rather than hard coding them.
    # LLMTask("task1", task1_prompt_hike).run(context)
    LLMTask("task1", task1_prompt_movie).run(context)

    # Step 2: LLM decides between task2 or task3
    next_task = decide_next_task(context)
    if next_task and next_task in tasks:
        LLMTask(next_task, task2_prompt).run(context)
    else:
        print("Invalid or no decision made by LLM. Ending workflow.")
    




def decide_next_task(context):
    task1_output = context.get("task1", "")
    prompt = f"""
    You are an AI workflow controller. Based on the output of task1 below, decide whether to run task2 or task3 next. Task2 is to go on hiking,
    Task3 is watching movie.1
    Output of task1:
    \"\"\"{task1_output}\"\"\"

    Respond with ONLY the task name: task2 or task3
    """

    response = ollama.chat(model='llama3.2', messages=[
        {"role": "user", "content": prompt}
    ])
    decision = response['message']['content'].strip().lower()
    print(f"[LLM Controller Decision] -> {decision}")
    return decision if decision in ["task2", "task3"] else None



# === Define Prompt Functions ===

def task1_prompt_movie(context):
    return "You are a decision assistant. Should I go hiking or watch a movie today? Take decisiom based on my physical health. I injured my leg in the gym today. Just explain your reasoning."
def task1_prompt_hike(context):
    return "You are a decision assistant. Should I go hiking or watch a movie today? I want to do something physically challenging today. Just explain your reasoning."
def task2_prompt(context):
    return "Great! You chose hiking. Suggest 3 hiking trails near Boulder, Colorado."

def task3_prompt(context):
    return "Nice! You chose watching a movie. Recommend 3 thought-provoking films released in the last 2 years."

# === Define Tasks ===

tasks = {
    "task1_hike": LLMTask("task1_hike", task1_prompt_hike),
    "task1_movie": LLMTask("task1_movie", task1_prompt_movie),
    "task2": LLMTask("task2", task2_prompt),
    "task3": LLMTask("task3", task3_prompt)
}