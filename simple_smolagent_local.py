#A very 'smol' smolagent.
#Runs locally (ran out of credits at HuggingFace :/) using ollama

from smolagents import ToolCallingAgent, LiteLLMModel, FinalAnswerTool, DuckDuckGoSearchTool

class App:
    def __init__(self):
         #Agent related
        self.agent = ToolCallingAgent(
            tools=[FinalAnswerTool(), DuckDuckGoSearchTool()],
            model = LiteLLMModel(model_id="ollama_chat/qwen2:0.5b",
                                 api_base="http://127.0.0.1:11434",
                                 num_ctx=8192),
            )

    def run(self):
        while True:
            query = input(">>")
            self.agent.run(query)


if __name__ == "__main__":
    App().run()




