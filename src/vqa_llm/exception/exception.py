import sys


class MyException(Exception):

    def __init__(self,
                 error_message:str,
                 error_details:sys):
        # self.error_message=error_message
        # _,_,exc_tb=error_details.exc_info()
        # print(exc_tb)

        # self.lineno=exc_tb.tb_lineno
        # self.file_name=exc_tb.tb_frame.f_code.co_filename
        super().__init__(error_message)
        self.error_message = error_message
        self.line_number = None
        self.file_name = None

        # Check if error_details is provided and get the traceback
        if error_details:
            _, _, exc_tb = error_details.exc_info()
            if exc_tb is not None:
                self.line_number = exc_tb.tb_lineno
                self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return f"Error occured in python script name [{self.file_name}] line number [{self.line_number}] error message [{str(self.error_message)}]"


if __name__=="__main__":
    try:
        a=1/0

    except Exception as e:
        #print(e)
        raise MyException(e,sys)