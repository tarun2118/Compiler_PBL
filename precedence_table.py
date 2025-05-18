from leading_last import get_lead_trail_sets

def build_precedence_table(gra, term_set, leadset, trailset):
    precedence_table = {}

    # Initialize table
    for term1 in term_set:
        precedence_table[term1] = {}
        for term2 in term_set:
            precedence_table[term1][term2] = " "

    # Build table from grammar productions
    for lhs in gra:
        for prod in gra[lhs]:
            for i in range(len(prod) - 1):
                a, b = prod[i], prod[i + 1]

                if a in term_set and b in term_set:
                    precedence_table[a][b] = "="
                if a in term_set and b not in term_set:
                    for c in leadset.get(b, []):
                        precedence_table[a][c] = "<"
                if a not in term_set and b in term_set:
                    for d in trailset.get(a, []):
                        precedence_table[d][b] = ">"

    # Add $ relations for start symbol
    start_symbol = list(gra.keys())[0]
    for a in leadset[start_symbol]:
        precedence_table['$'][a] = "<"
    for b in trailset[start_symbol]:
        precedence_table[b]['$'] = ">"
    precedence_table['$']['$'] = "="

    # Manual fix for associativity and precedence
    operators = ['+', '*']  # Define operators here

    for op in operators:
        precedence_table[op][op] = ">"  # Left associativity

    # Higher precedence for *
    precedence_table['+']['*'] = "<"
    precedence_table['*']['+'] = ">"

    return precedence_table

def get_all_data():
    gra, term_set, leadset, trailset = get_lead_trail_sets()
    precedence_table = build_precedence_table(gra, term_set, leadset, trailset)
    return gra, term_set, leadset, trailset, precedence_table
