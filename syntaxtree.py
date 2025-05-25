import json
from graphviz import Digraph

class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def get_precedence_table(output_json):
    with open(output_json, 'r') as f:
        data = json.load(f)
    return data['precedence_table']

def get_precedence(op1, op2, precedence_table):
    return precedence_table.get(op1, {}).get(op2, None)

def is_operator(c, precedence_table):
    # Force 'a' as terminal (operand)
    if c == 'a':
        return False
    return c in precedence_table

def build_syntax_tree(input_file, output_json):
    with open(input_file, 'r') as f:
        expr = f.read().strip()

    precedence_table = get_precedence_table(output_json)

    operator_stack = ['$']
    operand_stack = []

    def make_node(op):
        if len(operand_stack) < 2:
            raise Exception("Not enough operands")
        right = operand_stack.pop()
        left = operand_stack.pop()
        node = Node(op)
        node.left = left
        node.right = right
        operand_stack.append(node)

    expr += '$'
    i = 0
    while i < len(expr):
        symbol = expr[i]

        if symbol == ' ':
            i += 1
            continue

        if not is_operator(symbol, precedence_table):  # Operand
            operand_stack.append(Node(symbol))
            print(f"Pushed operand: {symbol}")
            i += 1
        else:
            while True:
                top = operator_stack[-1]

                # Explicit accept if both are end marker '$'
                if top == '$' and symbol == '$':
                    print("Both top and current symbol are end marker '$', accepting and finishing.")
                    i = len(expr)
                    break

                prec = get_precedence(top, symbol, precedence_table)
                print(f"Comparing top operator '{top}' with current symbol '{symbol}', precedence: {prec}")

                if prec == '<.':
                    operator_stack.append(symbol)
                    print(f"Pushed operator: {symbol}")
                    i += 1
                    break
                elif prec == '=.':
                    operator_stack.pop()
                    print(f"Popped operator for equal precedence: {top}")
                    i += 1
                    break
                elif prec == '>.':
                    op = operator_stack.pop()
                    print(f"Popped operator for reduction: {op}")
                    make_node(op)
                else:
                    raise Exception(f"Invalid precedence relation: {top} ? {symbol}")

    while operator_stack[-1] != '$':
        op = operator_stack.pop()
        print(f"Final reduction with operator: {op}")
        make_node(op)

    if len(operand_stack) != 1:
        raise Exception("Invalid expression: operand stack does not have exactly one element at the end")

    return operand_stack[0]

def draw_syntax_tree(root):
    dot = Digraph()

    def add_nodes_edges(node, parent_id=None):
        if node is None:
            return
        node_id = str(id(node))
        dot.node(node_id, node.value)
        if parent_id:
            dot.edge(parent_id, node_id)
        add_nodes_edges(node.left, node_id)
        add_nodes_edges(node.right, node_id)

    add_nodes_edges(root)
    return dot
