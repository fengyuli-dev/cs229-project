from typing import Optional

from llama_models.llama3.api.datatypes import SystemMessage, UserMessage
from llama_models.llama3.reference_impl.generation import Llama


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


if __name__ == "__main__":
    dialog = [
        SystemMessage(content="You must provide answer to arithmatic questions."),
        UserMessage(content="What is 2 + 2?"),
    ]
    llama = LlamaForInference(
        "/lfs/local/0/fengyuli/.cache/relgpt/.llama/checkpoints/Meta-Llama3.2-1B-Instruct"
    )
    print(llama.generate(dialog).content)
