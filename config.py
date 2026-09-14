# config.py
import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

# Get ApiKey from environment variable or use a default value
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
YIDONG_API_KEY = os.getenv("YIDONG_API_KEY")
YIDONG_API_KEY_1 = os.getenv("YIDONG_API_KEY_1")
YIDONG_API_KEY_1_NEW = os.getenv("YIDONG_API_KEY_1_NEW")
YIDONG_API_KEY_2 = os.getenv("YIDONG_API_KEY_2")
YIDONG_API_KEY_3 = os.getenv("YIDONG_API_KEY_3")
YIDONG_API_KEY_4 = os.getenv("YIDONG_API_KEY_4")


# set model parameters
def get_model_settings(model_name: str) -> dict:
    """
    Return model-specific settings
    """
    Setting_default = {
        "temperature": 0.0, #0.2
        "top_p": 1,
    }


    Setting_1 = {
        "temperature": 0.2,
        "top_p": 1,
    }

    model_map = {
        # "deepseek/deepseek-chat-v3.1:free": Setting_1,
        # "deepseek/deepseek-chat-v3.1": Setting_1,
        # "openai/gpt-5": Setting_1,
    }


    return model_map.get(model_name, Setting_default)