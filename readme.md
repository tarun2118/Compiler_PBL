# Operator Precedence Parser with Syntax Tree Visualizer

This project implements an **Operator Precedence Parser Tool** with syntax tree generation and visualization, built for educational and compiler construction purposes. It allows users to input grammar rules and arithmetic expressions, then visualizes the parsing process step-by-step using shift-reduce parsing logic and operator precedence tables.

## 🚀 Features

- ✅ Operator Precedence Table Construction (from grammar rules)
- ✅ Shift-Reduce Parsing using Precedence Comparison
- ✅ Real-Time Parsing Simulation (stack, input, action updates)
- ✅ Syntax Tree Generation from Reductions
- ✅ Syntax Tree Export using Graphviz (PNG format)
- ✅ Expression input from file (`input.txt`)
- ✅ Grammar and precedence loading from `output.json`

## 📁 Project Structure

├── main.cpp # Entry point for combined parsing logic
├── check_grammar.cpp # Grammar validation and processing
├── buildparsetable.cpp # Builds parse table from grammar
├── build_precedencetable.cpp # Generates operator precedence matrix
├── leadlast.cpp # Computes LEADING and TRAILING sets
├── syntaxtree.py # Builds and visualizes syntax tree
├── app.py # (Optional) GUI prototype (Tkinter)
├── input.txt # Expression to be parsed
├── output.json # Operator precedence table
├── outputparsing.json # Used during parsing steps
├── 2.txt # Temporary output or expression steps
