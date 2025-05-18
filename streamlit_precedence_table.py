# streamlit_precedence_table.py

import streamlit as st
from precedence_table import get_all_data

gra, term_set, leadset, trailset, precedence_table = get_all_data()

st.title("Operator Precedence Table Viewer")

st.header("Grammar")
for lhs, prods in gra.items():
    st.write(f"**{lhs}** → {' / '.join(prods)}")

st.header("Terminals")
st.write(", ".join(sorted(term_set)))

st.header("LEAD Sets")
for nt, leads in leadset.items():
    st.write(f"LEAD({nt}) = {{ {', '.join(sorted(leads))} }}")

st.header("TRAIL Sets")
for nt, trails in trailset.items():
    st.write(f"TRAIL({nt}) = {{ {', '.join(sorted(trails))} }}")

st.header("Operator Precedence Table")
term_list = sorted(term_set)
table_data = []

# Create header row
table_data.append([" "] + term_list)

# Create table rows
for row in term_list:
    row_data = [row]
    for col in term_list:
        row_data.append(precedence_table[row][col])
    table_data.append(row_data)

# Display as table
st.table(table_data)
