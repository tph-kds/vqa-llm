
from src.vqa_llm.logger  import logger
from src.vqa_llm.exception.exception import MyException
from src.vqa_llm.utils.file import File

class ProcessText:
    def __init__(self, name_file:str):
        self.name_file = name_file
        self.df = File(self.name_file).read_file_csv()
        self.labels_idx = self.labels_idx()

    def labels_idx(self):
        try:
            answer_vocab = list(set(self.df["answer"].tolist()))
            # # # Create a reverse mapping
            index_to_answer = {answer:i for i, answer in enumerate(answer_vocab)}
            logger.log_message("info", "Labels index created!")
            return index_to_answer

        except Exception as e:
            print(MyException("Error creating labels index", e))


    def inverse_labels_idx(self):
        try:
            logger.log_message("info", "Creating inverse labels index...")
            # Create an inverse dictionary
            inverse_labels = {v: k for k, v in self.labels_idx.items()}

            return inverse_labels
        
        except Exception as e:
            print(MyException("Error creating inverse labels index", e))