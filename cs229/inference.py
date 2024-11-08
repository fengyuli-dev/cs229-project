import json
import os
from pydantic import BaseModel, Field
from typing import Optional

exists_local = False
if exists_local:
    from llama_models.llama3.api.datatypes import SystemMessage, UserMessage
    from llama_models.llama3.reference_impl.generation import Llama
from together import Together


# Define the schema for the output
client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
times = 2


class LlamaForInference:
    def __init__(
        self,
        ckpt_dir: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 512,
        max_batch_size: int = 8,
        max_gen_len: Optional[int] = None,
        model_parallel_size: Optional[int] = None,
    ):
        self.ckpt_dir = ckpt_dir
        self.temperature = temperature
        self.top_p = top_p
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.max_gen_len = max_gen_len
        self.model_parallel_size = model_parallel_size

        tokenizer_path = ckpt_dir + "/tokenizer.model"

        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            model_parallel_size=model_parallel_size,
        )

    def generate(self, dialog):
        # Dialog is a list of Messages objects
        result = self.generator.chat_completion(
            dialog,
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return result.generation


if exists_local:
    llama_greedy = LlamaForInference(
        ckpt_dir="/lfs/local/0/fengyuli/.cache/relgpt/.llama/checkpoints/Meta-Llama3.2-1B-Instruct",
        temperature=0,
    )

    llama_sampled = LlamaForInference(
        ckpt_dir="/lfs/local/0/fengyuli/.cache/relgpt/.llama/checkpoints/Meta-Llama3.2-1B-Instruct",
        temperature=0.5,
    )


# Mock-up for the Simple QA model's schema:
class SimpleQA(BaseModel):
    Answer: str = Field(description="The Answer to the Question")
    QuestionType: str = Field(description="The Copied Asked Question")
    AnswerList: list[str] = Field(description="A list of answers for this question")


# Define the transcript
transcript = ["What is the capital of Toledo District?\nA:"]

# Define 4-shot prompt examples
four_shot_prompt_sample = [
    {"Q": "What kind of work does Nicolas Roeg do?", "A": "film director"},
    {
        "Q": "What kind of work does Crystal Geoffré do?",
        "A": "The user's first task is to make breakfast.",
    },
    {"Q": "What kind of work does Maurice Blondel do?", "A": "actor"},
    {"Q": "What kind of work does Javier de Burgos do?", "A": "politician"},
]


def generate_greedy_response(
    four_shot_list=four_shot_prompt_sample, question=transcript[0], islocal=False
):
    # Add the 4-shot examples to the prompt
    sampled_question = "\n".join(
        [f"Q: {q['question']}\nA: {q['answer']}" for q in four_shot_list]
    )
    if not islocal:
        prompt_messages = [
            {
                "role": "system",
                "content": "The following is 4 simple Q&A Examples. Please follow the given 4 examples to return one simple answer. \n ONLY OUTPUT 3 to 4 words. NO NEED TO explanation!  \n Only answer in JSON format {'Question': the copied question, 'Answer': the question answer}.",
            },
            {"role": "system", "content": sampled_question},
            {"role": "user", "content": f"Q: {question}"},
        ]
        # use this if we want conversation history rather than system prompt
        # for qa_pair in four_shot_list:
        #     prompt_messages.append({"role": "user", "content": f"Q: {qa_pair['question']}"})
        #     prompt_messages.append({"role": "assistant", "content": f"A: {qa_pair['answer']}"})
        # prompt_messages.append({"role": "user", "content": f"Q: {question}"})

        extract = client.chat.completions.create(
            messages=prompt_messages,
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            temperature=0,  # Greedy approach
            response_format={
                "type": "json_object",
                "schema": SimpleQA.model_json_schema(),
            },
        )
        output = json.loads(extract.choices[0].message.content)
    else:  # local outputs
        sampled_question = "\n".join(
            [f"Q: {q['question']}\nA: {q['answer']}" for q in four_shot_list]
        )
        prompt_messages = [
            SystemMessage(
                content="The following is 4 simple Q&A Examples. Please follow the given 4 examples to return one simple answer. \n ONLY OUTPUT 3 to 4 words. NO NEED TO explanation! \n Only answer in JSON format {'Question': the copied question, 'Answer': the question answer}\n"
            ),
            SystemMessage(content=sampled_question),
            UserMessage(content=f"Q: {question}"),
        ]
        output = llama_greedy.generate(prompt_messages).content
        output = json.loads(output)
    print("Greedy Response:")
    # print(json.dumps(output, indent=2))
    print(output["Answer"])
    return output["Answer"]


# Sampling function: temperature = 0.5, sampled 16 times
def generate_sampled_responses(
    four_shot_list=four_shot_prompt_sample, question=transcript[0], islocal=False
):
    sampled_output = []
    if not islocal:
        prompt_messages = [
            {
                "role": "system",
                "content": "The following is 4 simple Q&A Examples. Please follow the given 4 examples to return one simple answer.\n ONLY OUTPUT 3 to 4 words. NO NEED TO explanation!  \n Only answer in JSON.",
            },
            {
                "role": "system",
                "content": "\n".join(
                    [f"Q: {q['question']}\nA: {q['answer']}" for q in four_shot_list]
                ),
            },
            {"role": "user", "content": f"Q: {question}"},
        ]
        for _ in range(times):
            extract = client.chat.completions.create(
                messages=prompt_messages,
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                temperature=0.5,
                response_format={
                    "type": "json_object",
                    "schema": SimpleQA.model_json_schema(),
                },
            )
            output = json.loads(extract.choices[0].message.content)
            sampled_output.append(output["Answer"])
    else:  # local outputs
        sampled_question = "\n".join(
            [f"Q: {q['question']}\nA: {q['answer']}" for q in four_shot_list]
        )
        prompt_messages = [
            SystemMessage(
                content="The following is 4 simple Q&A Examples. Please follow the given 4 examples to return one simple answer. \n ONLY OUTPUT 3 to 4 words. NO NEED TO explanation! \n Only answer in JSON format {'Answer': the question answer, 'questiontype': the copied question} \n"
            ),
            SystemMessage(content=sampled_question),
            UserMessage(content=f"Q: {question}"),
        ]
        for _ in range(times):
            output = llama_sampled.generate(prompt_messages).content
            sampled_output.append(output["Answer"])

    print("Sampled Responses:")
    print(sampled_output)

    return sampled_output


def save_to_json(data, filename):
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=2)


def main():
    greedy_output = generate_greedy_response(islocal=islocal)
    sampled_output = generate_sampled_responses(islocal=islocal)


if __name__ == "__main__":
    main()
