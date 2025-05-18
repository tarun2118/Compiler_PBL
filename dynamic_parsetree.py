from graphviz import Digraph
import re

#this code will generate a parse tree using graphviz library
class ParseTreeVisualizer:
    def __init__(self):
        self.tree = Digraph("ParseTree")
        self.node_count = 0

    def add_node(self, label, parent=None):
        node_id = f"node{self.node_count}"
        self.tree.node(node_id, label)
        if parent:
            self.tree.edge(parent, node_id)
        self.node_count += 1
        return node_id

    def render(self, filename="parse_tree"):
        self.tree.render(filename=filename, format="png", cleanup=True)
        print(f"Parse tree saved as {filename}.png")

class GrammarParser:
    def __init__(self, grammar_file):
        self.rules = {}  # {nonterminal: [[symbols], [symbols], ...]}
        self.terminals = set()
        self.nonterminals = set()
        self.load_grammar(grammar_file)

    def load_grammar(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or '->' not in line:
                    continue
                lhs, rhs = line.split('->')
                lhs = lhs.strip()
                self.nonterminals.add(lhs)
                productions = rhs.split('/')
                prod_list = []
                for prod in productions:
                    prod = prod.strip()
                    symbols = prod.split()
                    prod_list.append(symbols)
                    # Collect terminals (anything not uppercase or '@')
                    for sym in symbols:
                        if sym != '@' and (not sym.isupper()):
                            self.terminals.add(sym)
                self.rules[lhs] = prod_list

        print("Loaded grammar:")
        for nt, prods in self.rules.items():
            print(f"{nt} -> {' / '.join([' '.join(p) for p in prods])}")
        print(f"Terminals: {sorted(self.terminals)}")
        print(f"Non-terminals: {sorted(self.nonterminals)}")

class RecursiveDescentParser:
    def __init__(self, grammar, tokens):
        self.grammar = grammar
        self.tokens = tokens
        self.pos = 0
        self.viz = ParseTreeVisualizer()

    def current_token(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def advance(self):
        self.pos += 1

    def parse_nonterminal(self, nonterminal, parent_node):
        # Create node for this nonterminal
        node = self.viz.add_node(nonterminal, parent_node)
        prods = self.grammar.rules.get(nonterminal, [])
        # Try each production in order
        for production in prods:
            save_pos = self.pos
            # print(f"Trying {nonterminal} -> {' '.join(production)} at pos {self.pos}")
            matched_nodes = []
            success = True
            # If production is epsilon '@', succeed immediately without consuming tokens
            if len(production) == 1 and production[0] == '@':
                # Epsilon production: add a special node maybe or just return
                epsilon_node = self.viz.add_node('@', node)
                return True, node

            for sym in production:
                if sym in self.grammar.nonterminals:
                    # parse nonterminal recursively
                    res, child_node = self.parse_nonterminal(sym, node)
                    if not res:
                        success = False
                        break
                    matched_nodes.append(child_node)
                else:
                    # sym is terminal
                    tok = self.current_token()
                    if tok == sym:
                        terminal_node = self.viz.add_node(sym, node)
                        self.advance()
                        matched_nodes.append(terminal_node)
                    else:
                        success = False
                        break
            if success:
                return True, node
            else:
                # backtrack
                self.pos = save_pos
                # remove children nodes added for this failed production (not strictly necessary)
        return False, None

    def parse(self, start_symbol):
        root = self.viz.add_node("Start")
        success, _ = self.parse_nonterminal(start_symbol, root)
        if success and self.pos == len(self.tokens):
            print("Parsing succeeded!")
            self.viz.render()
        else:
            print("Syntax Error: Parsing failed or extra tokens remaining.")

def tokenize(expr, terminals):
    # Tokenize by splitting on spaces or known terminals/operators
    # Let's split by spaces for now (you can make this more complex)
    expr = expr.strip()
    # For terminals like +, *, i, etc.
    pattern = '|'.join(re.escape(t) for t in sorted(terminals, key=lambda x: -len(x)))
    if pattern:
        tokens = re.findall(pattern, expr)
    else:
        tokens = expr.split()
    return tokens

if __name__ == "__main__":
    grammar_file = "norecursion.txt"
    grammar = GrammarParser(grammar_file)

    # Example input: "i + i * i"
    print(f"Enter expression using terminals {sorted(grammar.terminals)}:")
    expr = input()

    # Tokenize input based on terminals (you can improve tokenizer if needed)
    tokens = []
    i = 0
    expr = expr.replace(' ', '')
    # For your terminals like 'i', '+', '*', split accordingly
    # We'll match character by character for terminals
    terminals = grammar.terminals
    while i < len(expr):
        matched = False
        for t in sorted(terminals, key=lambda x: -len(x)):
            if expr.startswith(t, i):
                tokens.append(t)
                i += len(t)
                matched = True
                break
        if not matched:
            print(f"Unknown token starting at: {expr[i:]}")
            exit(1)

    print("Tokens:", tokens)

    parser = RecursiveDescentParser(grammar, tokens)
    # Start symbol is usually the first LHS in grammar file
    start_symbol = list(grammar.rules.keys())[0]
    parser.parse(start_symbol)
