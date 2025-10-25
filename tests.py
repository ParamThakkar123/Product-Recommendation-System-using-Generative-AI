import importlib
import sys
import types
import pytest

def _make_fake_streamlit(select_returns, text_inputs=None, file_upload=None):
    class DummySidebar:
        def __init__(self, returns):
            self.returns = returns
            self.idx = 0
        def selectbox(self, *args, **kwargs):
            if self.idx < len(self.returns):
                val = self.returns[self.idx]
            else:
                val = self.returns[-1]
            self.idx += 1
            return val
        def header(self, *a, **k):
            return None

    class FakeSt:
        def __init__(self):
            self.sidebar = DummySidebar(select_returns)
        def title(self, *a, **k): pass
        def write(self, *a, **k): pass
        def file_uploader(self, *a, **k): return file_upload
        def image(self, *a, **k): pass
        def text_input(self, prompt):
            return (text_inputs or {}).get(prompt, "")

    fake = FakeSt()
    mod = types.ModuleType("streamlit")
    mod.title = fake.title
    mod.sidebar = fake.sidebar
    mod.write = fake.write
    mod.file_uploader = fake.file_uploader
    mod.image = fake.image
    mod.text_input = fake.text_input
    return mod

def _install_minimal_fakes(select_returns, text_inputs=None, file_upload=None):
    # streamlit fake
    sys.modules['streamlit'] = _make_fake_streamlit(select_returns, text_inputs, file_upload)

    # google.generativeai fake
    genai = types.ModuleType("google.generativeai")
    def configure(api_key): pass
    class GenerativeModel:
        def __init__(self, *a, **k): pass
        def generate_content(self, *a, **k): return "[]" 
    genai.configure = configure
    genai.GenerativeModel = GenerativeModel
    sys.modules['google'] = types.ModuleType('google')
    sys.modules['google.generativeai'] = genai

    # dotenv fake
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *a, **k: None
    sys.modules['dotenv'] = dotenv

    # langchain_groq.chat_models fake
    chat_models = types.ModuleType("langchain_groq.chat_models")
    class ChatGroq:
        def __init__(self, *a, **k): pass
    chat_models.ChatGroq = ChatGroq
    sys.modules['langchain_groq'] = types.ModuleType('langchain_groq')
    sys.modules['langchain_groq.chat_models'] = chat_models

    # crewai fake
    crewai = types.ModuleType("crewai")
    class Agent:
        def __init__(self, *a, **k): pass
    class Task:
        def __init__(self, *a, **k): pass
    class Crew:
        def __init__(self, *a, **k):
            self.kickoff_called = False
        def kickoff(self):
            self.kickoff_called = True
            return {"mock": "result"}
    class Process:
        sequential = object()
    crewai.Agent = Agent
    crewai.Task = Task
    crewai.Crew = Crew
    crewai.Process = Process
    sys.modules['crewai'] = crewai

    # crewai_tools fake
    tools = types.ModuleType("crewai_tools")
    class SerperDevTool: pass
    class WebsiteSearchTool: pass
    tools.SerperDevTool = SerperDevTool
    tools.WebsiteSearchTool = WebsiteSearchTool
    sys.modules['crewai_tools'] = tools

    # langchain_google_genai fake
    lg = types.ModuleType("langchain_google_genai")
    class ChatGoogleGenerativeAI:
        def __init__(self, *a, **k): pass
    lg.ChatGoogleGenerativeAI = ChatGoogleGenerativeAI
    sys.modules['langchain_google_genai'] = lg

    # PIL.Image fake
    pil = types.ModuleType("PIL")
    class Image:
        @staticmethod
        def open(fp): return None
    pil.Image = Image
    sys.modules['PIL'] = pil
    sys.modules['PIL.Image'] = Image

def _import_main_with_fakes(select_returns, text_inputs=None, file_upload=None):
    _install_minimal_fakes(select_returns, text_inputs, file_upload)
    # ensure fresh import
    if 'main' in sys.modules:
        del sys.modules['main']
    return importlib.import_module('main')

def test_country_currency_mapping_present():
    """
    White-box: assert the internal mapping `country_currency` in [main.py](main.py)
    contains expected countries and currency codes.
    """
    m = _import_main_with_fakes(["United States", "Recommendation Model"])
    assert hasattr(m, "country_currency")
    assert m.country_currency["United States"] == "USD"
    assert m.country_currency["Eurozone"] == "EUR"
    assert m.country_currency["India"] == "INR"

def test_serper_api_key_set_on_import():
    """
    White-box: verify the module sets SERPER_API_KEY in the environment at import time.
    """
    m = _import_main_with_fakes(["United States", "Recommendation Model"])
    assert m.os.environ.get("SERPER_API_KEY") == "e5d1106caed5747b9774f708088544929d35117a"

def test_recommendation_flow_triggers_crew_kickoff():
    """
    White-box: simulate choosing 'Recommendation Model' and providing a text prompt.
    Ensure top-level variables `text_prompt` and `crew` are created and that Crew.kickoff()
    was invoked during module import.
    """
    text_inputs = {"Your Text Prompt": "camera"}
    m = _import_main_with_fakes(["India", "Recommendation Model"], text_inputs=text_inputs)
    # text_prompt is assigned at module level in main.py
    assert getattr(m, "text_prompt", None) == "camera"
    # crew should exist and be the fake Crew we injected
    assert hasattr(m, "crew")
    assert getattr(m.crew, "kickoff_called", False) is True

# New white-box tests appended below

def test_image_qa_sets_model_llm_and_response_when_file_uploaded():
    """
    White-box: simulate selecting 'Image Question Answering Model' with a file uploaded.
    Ensure the generative model and llm are created and response is produced.
    """
    fake_file = object()
    m = _import_main_with_fakes(["United States", "Image Question Answering Model"], file_upload=fake_file)
    # module should have created model, llm, uploaded_file and response variables
    assert hasattr(m, "model")
    assert hasattr(m, "llm")
    assert getattr(m, "uploaded_file", None) is not None
    # our fake GenerativeModel.generate_content returns "[]"
    assert getattr(m, "response", None) == "[]"

def test_web_search_flow_triggers_crew_kickoff_when_query_present():
    """
    White-box: simulate selecting 'Web Searching Model' and providing a search query.
    Ensure top-level `search_query` and `crew` are present and Crew.kickoff() ran.
    """
    text_inputs = {"Your Search Query": "smartphone"}
    m = _import_main_with_fakes(["Australia", "Web Searching Model"], text_inputs=text_inputs)
    assert getattr(m, "search_query", None) == "smartphone"
    assert hasattr(m, "crew")
    assert getattr(m.crew, "kickoff_called", False) is True

def test_recommendation_flow_without_prompt_does_not_kickoff():
    """
    White-box: when Recommendation Model is selected but no text prompt is provided,
    the Crew should be created but kickoff should not be called.
    """
    m = _import_main_with_fakes(["India", "Recommendation Model"], text_inputs={})
    assert hasattr(m, "crew")
    # kickoff should remain False because text_prompt is falsy
    assert getattr(m.crew, "kickoff_called", False) is False

def test_sidebar_selectbox_sequence_sets_selected_country_and_model_type():
    """
    White-box: confirm the selectbox selection order results in expected module attributes.
    """
    # first selectbox -> country, second selectbox -> model_type
    m = _import_main_with_fakes(["Eurozone", "Web Searching Model"], text_inputs={"Your Search Query": ""})
    assert getattr(m, "selected_country", None) == "Eurozone"
    assert getattr(m, "model_type", None) == "Web Searching Model"