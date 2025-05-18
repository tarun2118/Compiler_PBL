# error_handler.py

def handle_error(expression, message):
    return f"Syntax Error in '{expression}': {message}"

def check_errors(expression):
    if not expression.strip():
        return handle_error(expression, "Input is empty.")

    # Tokenization for parentheses check
    open_count = 0
    for ch in expression:
        if ch == '(':
            open_count += 1
        elif ch == ')':
            open_count -= 1
            if open_count < 0:
                return handle_error(expression, "Unmatched closing parenthesis.")
    if open_count > 0:
        return handle_error(expression, "Unmatched opening parenthesis.")

    # Add more error rules here if needed
    return None 