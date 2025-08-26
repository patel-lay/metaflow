import ollama
from metaflow import FlowSpec, step, IncludeFile, Parameter
from llmtask import LLMTask

class LLMController(FlowSpec):
    """
    A flow where Metaflow prints 'Hi'.

    Run this flow to validate that Metaflow is installed correctly.

    """
    @step
    def start(self):
        """
        This is the 'start' step. All flows must have a step named 'start' that
        is the first step in the flow.

        """
        print("AI Agent flow is starting.")
        self.next(self.task_1)
    
    @step
    def task_1(self):
        """
        A step for metaflow to introduce itself.

        """
        self.context = {}
        # Step 1: Run task1
        LLMTask("task1", task1_prompt_hike).run(self.context)
        # LLMTask("task1", task1_prompt_movie).run(self.context)

        # print(context)
        self.next_task = self.decide_next_task(self.context)

        self.next({"task2":self.task_2, "task3":self.task_3}, condition="next_task")

    @step
    def task_2(self):
        """
        A step for metaflow to introduce itself.

        """
        self.context = {}
        # Step 1: Run task1
        LLMTask("task2", task2_prompt).run(self.context)
        
        self.next(self.end)

    @step
    def task_3(self):
        """
        A step for metaflow to introduce itself.

        """
        self.context = {}
        # Step 1: Run task1
        LLMTask("task3", task3_prompt).run(self.context)
        
        self.next(self.end)

    @step
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.

        """
        print("LLMWorkflow is all done.")

    def decide_next_task(self, context):
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

# === Run Agent ===

if __name__ == "__main__":
    LLMController()