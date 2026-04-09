from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    model="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and city of a fictional person. \n {format instruction}",
    input_variables=[],
    partial_variables={"format instruction": parser.get_format_instructions()}
    # why is it called partial_variables ?
    # Because it is filled before RUN TIME only and not like the input_variables that are filled at runtime.
)

# prompt = template.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

chains = template | model | parser

final_result = chains.invoke({})
# Why {} ?
# By enforcing the rule that "everything takes a dictionary," the framework can blindly pass data from step to step. 
# If a step doesn't need the data, it just ignores the {} and runs anyway. 
# It prioritizes system predictability over typing a few less characters!
# The underlying code would become incredibly messy and prone to crashing if everytime we check if needs an argument or not

print(final_result)
print(type(final_result))