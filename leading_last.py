

def read_gra(fname):
    with open(fname, "r") as f:
        lines = f.read().strip().split('\n')
    first_line = lines[0].split()
    term_n = int(first_line[0])
    term_set = first_line[1:1 + term_n]

    gra = {}
    for line in lines[1:]:
        if not line.strip():
            continue
        lhs, rhs = line.split('->')
        lhs = lhs.strip()
        productions = [p.strip() for p in rhs.split('/')]
        gra[lhs] = productions

    return gra, term_set


def lead(nonterm, gra, arr=[], leadset={}):
    lead_ = set()
    for left_v in gra:
        if left_v == nonterm:
            for prod in gra[left_v]:
                if len(prod) == 1:
                    if prod in term_set:
                        lead_ |= {prod}
                    else:
                        if prod != nonterm and prod not in arr:
                            arr.append(prod)
                            lead_ |= lead(prod, gra, arr, leadset)
                        elif prod != nonterm and prod in leadset:
                            lead_ |= leadset[prod]
                else:
                    for i in prod:
                        if i in term_set:
                            lead_ |= {i}
                            if prod.index(i) != 0:
                                if prod[0] != nonterm and prod[0] not in arr:
                                    arr.append(prod[0])
                                    lead_ |= lead(prod[0], gra, arr, leadset)
                                elif prod[0] != nonterm and prod[0] in leadset:
                                    lead_ |= leadset[prod[0]]
                            break
    leadset[nonterm] = lead_
    return lead_


def trail(nonterm, gra, arr=[], trailset={}):
    trail_ = set()
    for left_v in gra:
        if left_v == nonterm:
            for prod in gra[left_v]:
                if len(prod) == 1:
                    if prod in term_set:
                        trail_ |= {prod}
                    else:
                        if prod != nonterm and prod not in arr:
                            arr.append(prod)
                            trail_ |= trail(prod, gra, arr, trailset)
                        elif prod != nonterm and prod in trailset:
                            trail_ |= trailset[prod]
                else:
                    for i in prod[::-1]:
                        if i in term_set:
                            trail_ |= {i}
                            if prod.index(i) != len(prod) - 1:
                                if prod[-1] != nonterm and prod[-1] not in arr:
                                    arr.append(prod[-1])
                                    trail_ |= trail(prod[-1], gra, arr, trailset)
                                elif prod[-1] != nonterm and prod[-1] in trailset:
                                    trail_ |= trailset[prod[-1]]
                            break
    trailset[nonterm] = trail_
    return trail_


def get_lead_trail_sets():
    global term_set
    gra, term_set = read_gra('grammar.txt')

    term_set.append('$') 

    leadset = {}
    trailset = {}

    for nt in gra:
        lead(nt, gra, [], leadset)
        trail(nt, gra, [], trailset)

    return gra, term_set, leadset, trailset

