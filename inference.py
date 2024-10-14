import json
import os
from together import Together
from pydantic import BaseModel, Field
import random



# Define the schema for the output
client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))
times = 16

# Mock-up for the Simple QA model's schema:
class SimpleQA(BaseModel):
    Answer: str = Field(description="The Answer to the Question")
    QuestionType: str = Field(description="The Copied Asked Question")
    AnswerList: list[str] = Field(
    description="A list of answers for this question"
    )

# Define the transcript
transcript = [
    "Q: What is the capital of Toledo District?\nA:"
]

# Define 4-shot prompt examples
four_shot_prompt_sample = [
    {
        "Q": "What kind of work does Nicolas Roeg do?",
        "A": "film director"
    },
    {
        "Q": "What kind of work does Crystal Geoffré do?",
        "A": "The user's first task is to make breakfast."
    },
    {
        "Q": " What kind of work does Maurice Blondel do?",
        "A": "actor"
    },
    {
        "Q": "What kind of work does Javier de Burgos do?",
        "A": "politician"
    }
]

def generate_greedy_response(four_shot_list=four_shot_prompt_sample, question_list=transcript):
    # Add the 4-shot examples to the prompt
    responses = []
    for question in question_list:
        prompt_messages = [
            {
                "role": "system",
                "content": "The following is 4 simple Q&A Examples. Plesae follow the given 4 examples to return one simple answer. \n No explanation needed. Only outputs 2-3 tokens. \n Only answer in JSON."
            },
            {
                "role": "system",
                "content": "\n".join([f"Q: {q['Q']}\nA: {q['A']}" for q in four_shot_list])
            },
            {
                "role": "user",
                "content": question
            },
        ]
        extract = client.chat.completions.create(
            messages=prompt_messages,
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            temperature=0,  # Greedy approach
            response_format={
                "type": "json_object",
                "schema": SimpleQA.model_json_schema(),
            }
        )
        output = json.loads(extract.choices[0].message.content)
        print("Greedy Response:")
        print(json.dumps(output, indent=2))
        responses.append(output)

    return output

# Sampling function: temperature = 16, sampled 16 times
def generate_sampled_responses(four_shot_list=four_shot_prompt_sample, question_list=transcript):
    responses = []
    for question in question_list:
        prompt_messages = [
            {
                "role": "system",
                "content": "The following is 4 simple Q&A Examples. Plesae follow the given 4 examples to return one simple answer. No explanation needed. Only response the answer. Only answer in JSON."
            },
            {
                "role": "system",
                "content": "\n".join([f"Q: {q['Q']}\nA: {q['A']}" for q in four_shot_list])
            },
            {
                "role": "user",
                "content": question
            },
        ]
        sampled_outputs = []
        for _ in range(times):
            extract = client.chat.completions.create(
                messages=prompt_messages,
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                temperature=0.5,
                response_format={
                    "type": "json_object",
                    "schema": SimpleQA.model_json_schema(),
                }
            )
            output = json.loads(extract.choices[0].message.content)
            sampled_outputs.append(output)

        print("Responses:")
        print(json.dumps(output, indent=2))
        responses.append(sampled_outputs)
    
    return responses

def save_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=2)

def main():
    greedy_output = generate_greedy_response()
    save_to_json(greedy_output, 'greedy_output.json')
    sampled_output = generate_sampled_responses()
    save_to_json(sampled_output, 'sampled_output.json')

if __name__ == "__main__":
    main()

