import tkinter as tk
from tkinter import messagebox, ttk

# Dummy function to simulate parser output (to be replaced by actual parser logic)
def dummy_parse_expression(expression):
    if not expression:
        raise ValueError("Expression is empty")
    return [
        {"stack": "$", "input": expression + "$", "action": "Shift"},
        {"stack": "$id", "input": "+id$", "action": "Shift"},
        {"stack": "$id+", "input": "id$", "action": "Shift"},
        {"stack": "$id+id", "input": "$", "action": "Reduce"},
        {"stack": "$E+E", "input": "$", "action": "Reduce"},
        {"stack": "$E", "input": "$", "action": "Accept"}
    ]

class OperatorPrecedenceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Operator Precedence Parser")
        self.root.geometry("800x600")
        self.root.resizable(False, False)

        # Store parsing steps
        self.parsing_steps = []
        self.current_step_index = 0

        # -------- Grammar Input Table --------
        grammar_label = ttk.Label(self.root, text="Enter Grammar Rules (LHS → RHS):", font=("Arial", 12))
        grammar_label.pack(pady=5)

        self.grammar_table = ttk.Treeview(self.root, columns=("LHS", "RHS"), show="headings", height=5)
        self.grammar_table.heading("LHS", text="LHS")
        self.grammar_table.heading("RHS", text="RHS")
        self.grammar_table.column("LHS", width=100, anchor=tk.CENTER)
        self.grammar_table.column("RHS", width=300, anchor=tk.CENTER)
        self.grammar_table.pack(pady=5)

        # Allow user to edit cells
        self.grammar_table.bind("<Double-1>", self.edit_grammar_cell)

        # Pre-fill with one row
        self.grammar_table.insert("", "end", values=("", ""))

        # -------- Expression Input --------
        self.input_entry = ttk.Entry(self.root, font=("Arial", 14), width=50)
        self.input_entry.pack(pady=10)

        # -------- Buttons --------
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=5)

        ttk.Button(button_frame, text="Add Rule", command=self.add_rule).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Parse", command=self.run_parser).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Next Step", command=self.show_next_step).pack(side=tk.LEFT, padx=5)


        # -------- Parsing Output Table --------
        output_label = ttk.Label(self.root, text="Parsing Steps:", font=("Arial", 12))
        output_label.pack(pady=5)

        columns = ("Stack", "Input", "Action")
        self.tree = ttk.Treeview(self.root, columns=columns, show="headings", height=10)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor=tk.CENTER, width=200)
        self.tree.pack(pady=10)

        self.tree.tag_configure("Shift", background="lightblue")
        self.tree.tag_configure("Reduce", background="lightgreen")
        self.tree.tag_configure("Accept", background="lightyellow")
        self.tree.tag_configure("Error", background="lightcoral")

    def edit_grammar_cell(self, event):
        item = self.grammar_table.identify_row(event.y)
        column = self.grammar_table.identify_column(event.x)
        if not item or not column:
            return
        x, y, width, height = self.grammar_table.bbox(item, column)
        entry_popup = tk.Entry(self.root)
        entry_popup.place(x=x + self.grammar_table.winfo_x(), y=y + self.grammar_table.winfo_y(), width=width, height=height)

        current_value = self.grammar_table.item(item, "values")[int(column[1]) - 1]
        entry_popup.insert(0, current_value)
        entry_popup.focus_set()

        def on_return(event):
            new_value = entry_popup.get()
            values = list(self.grammar_table.item(item, "values"))
            values[int(column[1]) - 1] = new_value
            self.grammar_table.item(item, values=values)
            entry_popup.destroy()

        entry_popup.bind("<Return>", on_return)
        entry_popup.bind("<FocusOut>", lambda e: entry_popup.destroy())

    def force_commit_edit(self):
        focus_widget = self.root.focus_get()
        if isinstance(focus_widget, tk.Entry) and str(focus_widget.master) == str(self.grammar_table):
            focus_widget.event_generate("<Return>")

    def run_parser(self):
        self.force_commit_edit()  # Ensures latest grammar edits are saved
        expression = self.input_entry.get().strip()
        if not expression:
            messagebox.showerror("Error", "Please enter an expression.")
            return

        # Retrieve grammar
        grammar = []
        for item in self.grammar_table.get_children():
            lhs, rhs = self.grammar_table.item(item)["values"]
            if lhs.strip() and rhs.strip():
                grammar.append((lhs.strip(), rhs.strip()))

        if not grammar:
            messagebox.showerror("Error", "Please enter grammar rules.")
            return

        self.clear_table()
        try:
            self.parsing_steps = dummy_parse_expression(expression)
            self.current_step_index = 0
            self.show_next_step()
        except Exception as e:
            messagebox.showerror("Parsing Error", str(e))

    def show_next_step(self):
        if self.current_step_index < len(self.parsing_steps):
            step = self.parsing_steps[self.current_step_index]
            row_id = self.tree.insert("", "end", values=(step["stack"], step["input"], step["action"]))
            self.tree.item(row_id, tags=(step["action"],))
            self.current_step_index += 1
        else:
            messagebox.showinfo("Parsing Complete", "All steps shown.")

    def clear_all(self):
        self.input_entry.delete(0, tk.END)
        self.clear_table()

    def clear_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def add_rule(self):
        self.grammar_table.insert("", "end", values=("", ""))


# Main driver
if __name__ == "__main__":
    root = tk.Tk()
    app = OperatorPrecedenceGUI(root)
    root.mainloop()
