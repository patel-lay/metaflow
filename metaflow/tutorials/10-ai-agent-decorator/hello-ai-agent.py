from metaflow import FlowSpec, step, IncludeFile, Parameter
from agentic_call import agentic_call

class LLMControlleredFlow(FlowSpec):
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

    @agentic_call   
    @step
    def task_1(self):
        """
        A step for metaflow to introduce itself.

        """
        print("Calling Agentic AI Decorator")

        self.next(self.end)

    @step
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.

        """
        print("LLMWorkflow is all done.")


# === Run Agent ===

if __name__ == "__main__":
    LLMControlleredFlow()