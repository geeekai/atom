from langchain_core.exceptions import OutputParserException
import time
import openai
from typing import Union, List
import numpy as np

class LangchainOutputParser:
    """
    A parser class for extracting and embedding information using Langchain and OpenAI APIs.
    """
    
    def __init__(self, llm_model, embeddings_model, sleep_time: int = 5) -> None:
        """
        Initialize the LangchainOutputParser with specified API key, models, and operational parameters.
        
        Args:
        api_key (str): The API key for accessing OpenAI services.
        embeddings_model_name (str): The model name for text embeddings.
        model_name (str): The model name for the Chat API.
        temperature (float): The temperature setting for the Chat API's responses.
        sleep_time (int): The time to wait (in seconds) when encountering rate limits or errors.
        """
        #self.model = ChatOpenAI(api_key=api_key, model_name=model_name, temperature=temperature)
        #self.embeddings_model = OpenAIEmbeddings(model=embeddings_model_name, api_key=api_key)
        
        self.model = llm_model
        self.embeddings_model = embeddings_model
        self.sleep_time = sleep_time

    async def calculate_embeddings(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Calculate embeddings for the given text using the initialized embeddings model.
        
        Args:
        text (Union[str, List[str]]): The text or list of texts to embed.
        
        Returns:
        np.ndarray: The calculated embeddings as a NumPy array.
        
        Raises:
        TypeError: If the input text is neither a string nor a list of strings.
        """
        if isinstance(text, list):
            embeddings = await self.embeddings_model.aembed_documents(text)
        elif isinstance(text, str):
            embeddings = await self.embeddings_model.aembed_query(text)
        else:
            raise TypeError("Invalid text type, please provide a string or a list of strings.")
        return np.array(embeddings)
    
    async def extract_information_as_json_for_context(self,
                                                 output_data_structure,
                                                 contexts: List[str],
                                                 IE_query: str = '''
                                                    # DIRECTIVES :
                                                    - Act like an experienced information extractor.
                                                    - If you do not find the right information, keep its place empty.
                                                    '''):
        
        structured_llm = self.model.with_structured_output(output_data_structure)

        async def batch_distillation():
            # Prepare a list of prompts to be processed in batch
            prompts = [
                f"# Context: {context}\n\n# Question: {IE_query}\n\nAnswer: "
                for context in contexts
            ]

            # Process the prompts in batch
            try:
                outputs = await structured_llm.abatch(prompts)
                return outputs
            except openai.BadRequestError as e:
                print(f"Too much requests, we are sleeping! \n the error is {e}")
                time.sleep(self.sleep_time)
                return self.extract_information_as_json_for_context(contexts=contexts, 
                                                                    output_data_structure=output_data_structure,
                                                                    IE_query=IE_query)

            except openai.RateLimitError as e:
                print("Too much requests exceeding rate limit, we are sleeping!", e )
                time.sleep(self.sleep_time)
                return self.extract_information_as_json_for_context(contexts=contexts, 
                                                                    output_data_structure=output_data_structure,
                                                                    IE_query=IE_query)
                
            except OutputParserException:
                print("Error in parsing the context")
                pass
            
        
        return await batch_distillation()
