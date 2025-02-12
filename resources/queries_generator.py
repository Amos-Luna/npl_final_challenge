import os
from typing import List
from langchain_openai import ChatOpenAI
from resources.schemas import OutputQuerySchema
from resources.prompt_templates import QueryRewriterTemplate
from langchain.prompts import PromptTemplate


class SimilQuerysGenerator:
    """Generate new querys related to original"""
    def __init__(
        self, 
        num_queries: int,
        user_query: str
    ) -> None:
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL")
        self.model = ChatOpenAI(
            api_key= self.openai_api_key, 
            model= self.openai_model,
            temperature= 0.8,
            top_p= 0.9
        )
        self.model_improved = self.model.with_structured_output(OutputQuerySchema)
        self.num_queries = num_queries
        self.user_query = user_query
        
        
    def get_new_querys(
        self
    )-> List[str]:
        """Genration of new custom querys related to original"""
        prompt = PromptTemplate(
            template = f"{QueryRewriterTemplate.system_prompt.template} \n {QueryRewriterTemplate.user_prompt.template}",
            input_variables=["num_queries", "user_query"]
        )
        chain = prompt | self.model_improved
        try: 
            response: List[str] = chain.invoke(
                {
                    "num_queries": self.num_queries,
                    "user_query": self.user_query
                }
            )
            return response.new_querys
        except Exception as e:
            print(f"[Error] --> {e}")