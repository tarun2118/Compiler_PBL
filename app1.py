import streamlit as st
import subprocess
import json
import pandas as pd
import os
from syntaxtree import build_syntax_tree, draw_syntax_tree

st.title("Operator Precedence Grammar Parser and Syntax Tree Builder")

# Initialize session state variables
if 'grammar_processed' not in st.session_state:
    st.session_state.grammar_processed = False
if 'grammar_text' not in st.session_state:
    st.session_state.grammar_text = ""
if 'grammar_data' not in st.session_state:
    st.session_state.grammar_data = None

if 'parsing_completed' not in st.session_state:
    st.session_state.parsing_completed = False
if 'parse_step' not in st.session_state:
    st.session_state.parse_step = 0

if 'syntax_tree_ready' not in st.session_state:
    st.session_state.syntax_tree_ready = False
if 'syntax_tree' not in st.session_state:
    st.session_state.syntax_tree = None

# --- Step 1: Grammar Input and Processing ---
grammar_text = st.text_area("Enter your grammar here:", value=st.session_state.grammar_text)

if st.button("Process Grammar"):
    if grammar_text.strip() == "":
        st.error("Please enter some grammar rules!")
    else:
        with open("2.txt", "w") as f:
            f.write(grammar_text)
        st.success("Grammar saved to 2.txt")

        # Run C++ grammar analyzer
        result = subprocess.run(["check.exe", "2.txt"], capture_output=True, text=True)

        if result.returncode != 0:
            st.error(f"Error running C++ program:\n{result.stderr}")
            st.session_state.grammar_processed = False
        else:
            st.success("Grammar processed successfully.")
            try:
                with open("output.json", "r") as f:
                    data = json.load(f)
                st.session_state.grammar_data = data
                st.session_state.grammar_processed = True
                st.session_state.grammar_text = grammar_text

                # Reset parsing and syntax tree states
                st.session_state.parsing_completed = False
                st.session_state.parse_step = 0
                st.session_state.syntax_tree_ready = False
                st.session_state.syntax_tree = None

            except Exception as e:
                st.error(f"Error loading output.json: {e}")
                st.session_state.grammar_processed = False

# --- Step 2: Show Grammar Info if processed ---
if st.session_state.grammar_processed and st.session_state.grammar_data:
    data = st.session_state.grammar_data

    st.header("Grammar Rules")
    for rule in data["grammar"]:
        st.text(rule)

    if data.get("is_valid", False):
        st.success("Grammar is a valid Operator Precedence Grammar.")
    else:
        st.error("Grammar is NOT a valid Operator Precedence Grammar.")
        st.stop()

    st.write(f"**Start Symbol:** {data['start_symbol']}")

    st.header("LEAD and LAST Sets")
    for nt, sets in data["lead_last"].items():
        lead = ", ".join(sets["lead"])
        last = ", ".join(sets["last"])
        st.markdown(f"**{nt}** → LEAD: {{ {lead} }}, LAST: {{ {last} }}")

    st.header("Precedence Table")
    table = data["precedence_table"]
    row_keys = list(table.keys())
    col_keys = set()
    for inner in table.values():
        col_keys.update(inner.keys())
    col_keys = sorted(col_keys)
    table_data = []
    for row in row_keys:
        row_data = [table.get(row, {}).get(col, "") for col in col_keys]
        table_data.append(row_data)
    df = pd.DataFrame(table_data, index=row_keys, columns=col_keys)
    st.dataframe(df.style.set_properties(**{'text-align': 'center'}))

    # --- Step 3: Input string for parsing ---
    st.header("Input String for Parsing")

    with st.form("parse_form"):
        input_string = st.text_input("Enter the input string to parse:")
        submitted = st.form_submit_button("Parse Input String")

    if submitted:
        if input_string.strip() == "":
            st.error("Please enter an input string!")
        else:
            with open("input.txt", "w") as f:
                f.write(input_string)
            st.success("Input string saved to input.txt")

            # Run your parser (C++ program) on input string
            result = subprocess.run(["check.exe", "input.txt"], capture_output=True, text=True)

            if result.returncode != 0:
                st.error(f"Error running parser:\n{result.stderr}")
                st.session_state.parsing_completed = False
            else:
                st.success("Parsing completed successfully.")
                st.session_state.parsing_completed = True
                st.session_state.parse_step = 0
                st.session_state.syntax_tree_ready = False
                st.session_state.syntax_tree = None

    # --- Step 4: Show Parsing Steps ---
    if st.session_state.parsing_completed:
        if os.path.exists("outputparsing.json"):
            with open("outputparsing.json", "r") as f:
                trace_data = json.load(f)

            st.header("Parsing Steps")

            max_step = len(trace_data) - 1
            step_to_show = st.session_state.parse_step

            # Display all steps up to current step
            for i in range(step_to_show + 1):
                step = trace_data[i]
                st.write(f"**Step {i + 1}:**")
                st.write(f"Stack: {step['stack']}")
                st.write(f"Input Symbol: {step['input']}")
                st.write(f"Action: {step['action']}")
                st.markdown("---")

            # Navigation buttons for parsing steps
            cols = st.columns(3)
            if cols[0].button("Previous") and step_to_show > 0:
                st.session_state.parse_step -= 1
            if cols[1].button("Next") and step_to_show < max_step:
                st.session_state.parse_step += 1
            if cols[2].button("Finish Parsing"):
                st.session_state.parse_step = max_step

            if step_to_show == max_step:
                st.success("Parsing completed. You can now build the syntax tree below.")
                st.session_state.syntax_tree_ready = False

    # --- Step 5: Input string for syntax tree & build syntax tree ---
    if st.session_state.parsing_completed and st.session_state.parse_step == (len(trace_data) - 1):
        st.header("Build Syntax Tree")

        with st.form("syntax_tree_form"):
            syntax_input = st.text_input("Enter the input string to build syntax tree:", value=input_string)
            build_submitted = st.form_submit_button("Build Syntax Tree")

        if build_submitted:
            if syntax_input.strip() == "":
                st.error("Please enter an input string to build syntax tree!")
            else:
                with open("input.txt", "w") as f:
                    f.write(syntax_input)
                st.success("Input string saved to input.txt for syntax tree")

                try:
                    root = build_syntax_tree("input.txt", "output.json")
                    if root:
                        st.session_state.syntax_tree_ready = True
                        st.session_state.syntax_tree = root
                        st.success("Syntax tree built successfully.")
                    else:
                        st.error("Failed to build syntax tree.")
                except Exception as e:
                    st.error(f"Error building syntax tree: {e}")
                    st.session_state.syntax_tree_ready = False

    # --- Step 6: Display syntax tree ---
    if st.session_state.syntax_tree_ready:
        st.header("Syntax Tree Visualization")
        dot = draw_syntax_tree(st.session_state.syntax_tree)
        st.graphviz_chart(dot.source)

else:
    st.info("Please enter and process your grammar first.")
