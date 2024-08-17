
from src.vqa_llm.logger  import logger
from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.utils.file import File

class ProcessText:
    def __init__(self, name_file:str):
        self.name_file = name_file
        self.df = File(self.name_file).read_file_csv()

    def labels_idx(self):
        try:
            answer_vocab = list(set(self.df["answer"].tolist()))
            # # # Create a reverse mapping
            index_to_answer = {answer:i for i, answer in enumerate(answer_vocab)}
            logger.log_message("info", "Labels index created!")
            return index_to_answer

        except Exception as e:
            print(MyException("Error creating labels index", e))