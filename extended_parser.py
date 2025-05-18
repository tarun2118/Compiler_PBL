# extended_parser.py

from error_handler import check_errors

def tokenize(expression):
    import re
    tokens = []
    token_specification = [
        ('NUMBER',   r'\d+(\.\d*)?'),
        ('IDENT',    r'[a-zA-Z_]\w*'),
        ('OP',       r'[+\-*/()]'),
        ('SKIP',     r'[ \t]+'),
        ('MISMATCH', r'.'),
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in token_specification)
    for mo in re.finditer(tok_regex, expression):
        kind = mo.lastgroup
        value = mo.group()
        if kind in ['NUMBER', 'IDENT', 'OP']:
            tokens.append(value)
        elif kind == 'SKIP':
            continue
        elif kind == 'MISMATCH':
            raise SyntaxError(f"Unexpected character: {value}")
    return tokens

def parse_expression(expression):
    # Step 1: Validate input and tokenize
    error_msg = check_errors(expression)
    if error_msg:
        return error_msg

    tokens = tokenize(expression)

    # Step 2: [Placeholder] Implement operator precedence parsing logic
    # For now, just show the tokens
    return f"Valid Expression. Tokens: {tokens}"

# Example usage
if __name__ == "_main_":
    expr = input("Enter an expression: ")
    result = parse_expression(expr)
    print(result)