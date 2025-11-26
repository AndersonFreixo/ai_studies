#A very 'smol' smolagent.
#Runs locally using ollama or through HF serverless API

from smolagents import ToolCallingAgent,FinalAnswerTool, DuckDuckGoSearchTool, InferenceClientModel, LiteLLMModel
from custom_tools import get_weather
import dotenv, argparse

class App:
    def __init__(self, is_local):
        #Agent related
        remote_success = False
        #If local flag is not set, try to get model from HuggingFace hub
        if is_local:
            model = LiteLLMModel(model_id="ollama_chat/qwen2:7b",
                                    api_base="http://127.0.0.1:11434",
                                    num_ctx=12000)
        #If can't get from server, try to connect to ollama local server
        else:
            dotenv.load_dotenv() #loads HF_TOKEN
            model = InferenceClientModel()

        self.agent = ToolCallingAgent(
            tools=[get_weather, FinalAnswerTool(), DuckDuckGoSearchTool()],
            model = model,
            )

    def run(self):
        while True:
            query = input(">>")
            self.agent.run(query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Simple Smolagent",
        description="A simple AI agent implementation using smolagents")
    parser.add_argument('-l',
                        '--local',
                        action="store_true",
                        help="Run agent locally with Ollama")
    args = parser.parse_args()
    App(args.local).run()




