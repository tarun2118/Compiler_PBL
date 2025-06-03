#include <iostream>
#include <stack>
#include <vector>
#include <map>
#include <string>
#include <tuple>
#include <fstream>
#include <algorithm>
using namespace std;

// Helper: convert full stack contents to a space-separated string (bottom to top)
string stackToString(const stack<string>& s) {
    stack<string> temp = s;
    vector<string> elems;
    while (!temp.empty()) {
        elems.push_back(temp.top());
        temp.pop();
    }
    reverse(elems.begin(), elems.end());
    string res = "";
    for (auto& e : elems) res += e + " ";
    if (!res.empty()) res.pop_back(); // remove trailing space
    return res;
}

// Parse input using operator precedence table passed as argument
vector<tuple<string, string, string>> buildParseTable(
    const map<string, map<string, string>>& precedenceTable,
    const string& inputString
) {
    vector<tuple<string, string, string>> trace;
    stack<string> parseStack;
    string buffer = inputString + "$";
    parseStack.push("$");

    int pointer = 0;
    while (true) {
        // Find topmost terminal in stack
        string topTerminal = "";
        stack<string> tempStack = parseStack;
        while (!tempStack.empty()) {
            string top = tempStack.top();
            if (precedenceTable.count(top)) {
                topTerminal = top;
                break;
            }
            tempStack.pop();
        }

        if (pointer >= (int)buffer.size()) {
            trace.push_back({"ERROR", "$", "Unexpected end of input"});
            break;
        }

        string currentInput(1, buffer[pointer]);

        if (topTerminal.empty() || precedenceTable.at(topTerminal).count(currentInput) == 0) {
            trace.push_back({"ERROR", buffer.substr(pointer), "ERROR: No rule"});
            break;
        }

        string relation = precedenceTable.at(topTerminal).at(currentInput);

        if (relation == "<." || relation == "=.") {
            // Shift
            parseStack.push(currentInput);
            pointer++;
            trace.push_back({"(Shift) Stack: " + stackToString(parseStack), buffer.substr(pointer), "Shift " + currentInput});
        } else if (relation == ">.") {
            // Reduce: pop until a terminal with relation <. to popped symbol found
            vector<string> toReduce;
            bool found = false;

            while (!parseStack.empty()) {
                string sym = parseStack.top();
                parseStack.pop();
                toReduce.push_back(sym);

                if (!parseStack.empty()) {
                    string prev = parseStack.top();
                    if (precedenceTable.count(prev) &&
                        precedenceTable.at(prev).count(sym) &&
                        precedenceTable.at(prev).at(sym) == "<.") {
                        found = true;
                        break;
                    }
                } else {
                    break;
                }
            }

            if (!found) {
                trace.push_back({"ERROR", buffer.substr(pointer), "ERROR: Invalid reduce"});
                break;
            }

            trace.push_back({"(Reduce) Stack: " + stackToString(parseStack), buffer.substr(pointer), "Reduce: " + to_string(toReduce.size()) + " symbols"});
        } else {
            trace.push_back({"ERROR", buffer.substr(pointer), "Invalid relation"});
            break;
        }

        // Accept condition: stack only has '$' and input is '$'
        if (parseStack.size() == 1 && parseStack.top() == "$" && pointer == (int)buffer.size() - 1 && buffer[pointer] == '$') {
            trace.push_back({"$", "$", "Accept"});
            break;
        }
    }

    return trace;
}

// Save trace to JSON file (can be reused as is)
void saveTraceToJSON(const vector<tuple<string, string, string>>& trace, const string& filename) {
    ofstream out(filename);
    out << "[\n";
    for (size_t i = 0; i < trace.size(); ++i) {
        string stack_str = get<0>(trace[i]);
        string input_str = get<1>(trace[i]);
        string action_str = get<2>(trace[i]);
        out << "  {\n";
        out << "    \"stack\": \"" << stack_str << "\",\n";
        out << "    \"input\": \"" << input_str << "\",\n";
        out << "    \"action\": \"" << action_str << "\"\n";
        out << "  }";
        if (i != trace.size() - 1) out << ",";
        out << "\n";
    }
    out << "]\n";
    out.close();
}
