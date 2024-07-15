import sys
import re

log_mode = True
border_text = "■" + " "*75 + "\n"
# Handles print statements.
def log(stmt: any):
    if log_mode:
        print(stmt)
    else:
        pass

# Simple border for seperating terminal logs.
def border(stmt: any = '') -> None:
    log(f"\n{stmt}{border_text}\n")

# Cleans data type output.
def strip_data_type(arg_string: str) -> str:
    arg_string: str = str(arg_string)[1:len(str(arg_string))-1].replace("'","").replace("class","").replace("<","").replace(">","").replace("__main__.","")
    arg_string = re.sub(r'\s+',' ',arg_string)        
    return arg_string

# Class for logging method details.
class MethodLog:

    def __init__(self):
        self.method_logs_count: int = 0

    def increment(self) -> None:
        self.prev_method_log_count = self.method_logs_count
        self.method_logs_count += 1

    # Logs the actual data type of the arguments being passed, rather than the ones suggested to be passed.
    def start(self, func_name: str, args: dict):
        arg_string: str = strip_data_type(args)
        arg_string_dynamic: str = arg_string if len(arg_string) < len(border_text)-30 else f"{arg_string[:len(border_text)-30]} ...\n           -{arg_string[len(border_text)-30:]}" if len(arg_string) > len(border_text)-30 else "Arguments have not been detailed."

        self.increment()
        border(f"■ {self.method_logs_count} ")
        log(f"METHOD     {func_name}({arg_string_dynamic})")

    # Logs function's return status, return data type, and return memory size(KB).
    def end(self, status: str, return_val: any):
        status_case: str = "Success" if status == 1 else "Failed" if status == 0 else "Unset Status"
        return_type: str = strip_data_type(str(type(return_val)))
        return_size: str = str(sys.getsizeof(return_val)*10**(-3))+"KB"

        log(f"RETURN     Status: {status_case} | Type:{return_type} | Size: {return_size}\n")

# Global Instance
ML = MethodLog()