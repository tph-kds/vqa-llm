# a original struture json file with questions attribution

questions = {
    "info" : info,
    "task_type" : str,
    "data_type": str,
    "data_subtype": str,
    "questions" : [question],
    "license" : license
    }

    
    
    question = {
        "question_id" : int,
        "image_id" : int,
        "question" : str
        }


# a original struture json file with answers attribution
answers = {
        "info" : info,
        "data_type": str,
        "data_subtype": str,
        "annotations" : [annotation],
        "license" : license
        }

        annotation = {
            "question_id" : int,
            "image_id" : int,
            "question_type" : str,
            "answer_type" : str,
            "answers" : [answer],
            "multiple_choice_answer" : str
            }
            
            answer = {
            "answer_id" : int,
            "answer" : str,
            "answer_confidence": str
            }


# a main struture for using in this project
use_main = {
    "question_id" : int,
    "image_id" : int,
    "question" : str,
    "answer" : str
}