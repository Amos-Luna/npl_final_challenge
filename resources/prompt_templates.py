from typing import ClassVar
from langchain.prompts import PromptTemplate


class QueryRewriterTemplate:
    system_prompt: ClassVar[PromptTemplate] = PromptTemplate.from_template(
        """Generate {num_queries} paraphrased versions [IN SPANISH LANGUAGE] of the following question while keeping the original meaning intact."""
    )
    user_prompt: ClassVar[PromptTemplate] = PromptTemplate.from_template(
        """Paraphrase this question {num_queries} times: {user_query}"""
    )