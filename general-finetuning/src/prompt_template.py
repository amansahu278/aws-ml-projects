from dataclasses import dataclass
from re import S
from turtle import st

@dataclass
class PromptTemplate:
    name: str
    instruction_tag: str
    input_tag: str
    response_tag: str

    def format(self, instruction: str, input_text: str, response: str):
        if input_text.strip():
            return (
                f"{self.instruction_tag}\n{instruction.strip()}\n"
                f"{self.input_tag}\n{input_text.strip()}\n"
                f"{self.response_tag}\n{response.strip()}\n"
            )
        else:
            return (
                f"{self.instruction_tag}\n{instruction.strip()}\n"
                f"{self.response_tag}\n{response.strip()}\n"
            )
    @classmethod
    def from_name(cls, name: str):

        if name.lower() == "alpaca":
            return cls(
                name="alpaca",
                instruction_tag="### Instruction:",
                input_tag="### Input:",
                response_tag="### Response:"
            )
        elif name.lower() == "vicuna":
            return cls(
                name="vicuna",
                instruction_tag="### Human:",
                input_tag="",
                response_tag="### Assistant:"
            )
        else:
            raise ValueError(f"Prompt template {name} not found")