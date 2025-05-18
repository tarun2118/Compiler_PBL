This is my Compiler PBL Project
The objective of this project is to develop an Operator Precedence Parser Tool using Python that allows users—especially students learning compiler design—to visualize the parsing of arithmetic expressions interactively. Parsing is a core concept in compilers, and this tool transforms theoretical parsing mechanics into an engaging and understandable form.
The system accepts arithmetic expressions involving operators such as +, -, *, /, ^, and parentheses. It uses regular expression-based tokenization, precedence comparison logic, and a shift-reduce parser simulation to interpret the expressions. At every parsing step, the tool displays the current operator stack, output, and actions taken (e.g., shift or reduce), helping users follow how an expression is evaluated and converted into its postfix form.
The tool is built with a Tkinter-based GUI that enables users to:
•	Enter grammar rules and expressions,
•	View step-by-step parsing actions in a tabular format,
•	See parsing errors such as unmatched parentheses or invalid tokens.
Key features include:
•	Grammar input for custom parsing logic,
•	A dynamically updating parser state display,
•	Error handling for better learning feedback,
•	And visual output generation using Graphviz for parse tree illustration.
This project enhances educational understanding of operator precedence parsing and compiler front-end design, bridging the gap between theory and practical implementation. It is especially useful for computer science students studying syntax analysis, language processing, or compiler construction.
1. System Design & Architecture
•	Modular Code Structure:
o	parser.py: Handles shift-reduce logic, operator precedence, and parse tree construction.
o	visualizer.py: Uses Graphviz to dynamically generate and export parse trees.
o	gui.py: Tkinter-based interface for user input and step-by-step parser display.
o	extended_parser.py: Combines tokenizer, function calls, and expression evaluation.
o	error_handler.py: Catches syntax errors (e.g., unmatched parentheses, invalid tokens).
•	Control Flow:
1.	User inputs expression via GUI.
2.	Input is tokenized using regular expressions.
3.	Tokens are parsed using shift-reduce logic based on an operator precedence table.
4.	Parsing steps (stack, input, action) are updated live in a GUI table.
5.	If no error, final parse tree is rendered and saved using Graphviz.
____________
2. Communication & Data Flow
•	Internal function calls between modules (no networking involved).
•	GUI calls parsing and visualization modules directly.
•	Parsing state (stack, input, output) is passed as structured data for display.
________________________________________
3. Libraries Used
•	Tkinter: For GUI creation and event handling.
•	Graphviz: For drawing and exporting parse trees.
•	re (Regular Expressions): For lexical analysis (tokenization).
•	ttk: For modern table widgets in the GUI.
________________________________________
4. Scalability
•	Designed to allow future upgrades, such as:
o	Support for additional operators
o	Grammar rule customization
o	More advanced parser types