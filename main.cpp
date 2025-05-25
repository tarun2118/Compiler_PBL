#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <iomanip>
using namespace std;

bool isOperatorPrecedenceGrammar(const string &filename);
unordered_map<string, vector<vector<string>>> calculateLeadLast(const string &filename);
map<string, map<string, string>> buildPrecedenceTableFromGrammar(const string &filename, const unordered_map<string, vector<vector<string>>> &leadLastMap, const string &startSymbol);
vector<tuple<string, string, string>> buildParseTable(const map<string, map<string, 
    string>>& precedenceTable, const string& inputString);
void saveTraceToJSON(const vector<tuple<string, string, string>>& trace, const string& filename);


string getStartSymbol(const string &filename) {
    ifstream file(filename);
    char ch;
    while (file.get(ch)) {
        if (!isspace(ch)) {
            return string(1, ch);
        }
    }
    return "";
}

vector<string> readGrammar(const string &filename) {
    vector<string> grammar;
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        if (!line.empty())
            grammar.push_back(line);
    }
    return grammar;
}

int main() {
    string filename = "2.txt";

    // Output JSON file
    ofstream jsonFile("output.json");
    jsonFile << "{\n";

    // Read and print grammar
    vector<string> grammar = readGrammar(filename);
    jsonFile << "  \"grammar\": [\n";
    for (size_t i = 0; i < grammar.size(); ++i) {
        jsonFile << "    \"" << grammar[i] << "\"";
        if (i != grammar.size() - 1) jsonFile << ",";
        jsonFile << "\n";
    }
    jsonFile << "  ],\n";

    // Grammar validity
    bool valid = isOperatorPrecedenceGrammar(filename);
    jsonFile << "  \"is_valid\": " << (valid ? "true" : "false") << ",\n";

    if (!valid) {
        jsonFile << "  \"message\": \"Grammar is not valid.\"\n}";
        jsonFile.close();
        return 1;
    }

    string startSymbol = getStartSymbol(filename);
    jsonFile << "  \"start_symbol\": \"" << startSymbol << "\",\n";

    auto leadLastMap = calculateLeadLast(filename);
    jsonFile << "  \"lead_last\": {\n";
    size_t count = 0;
    for (const auto &entry : leadLastMap) {
        jsonFile << "    \"" << entry.first << "\": {\n";
        // LEAD
        jsonFile << "      \"lead\": [";
        for (size_t i = 0; i < entry.second[0].size(); ++i) {
            jsonFile << "\"" << entry.second[0][i] << "\"";
            if (i != entry.second[0].size() - 1) jsonFile << ", ";
        }
        jsonFile << "],\n";

        // LAST
        jsonFile << "      \"last\": [";
        for (size_t i = 0; i < entry.second[1].size(); ++i) {
            jsonFile << "\"" << entry.second[1][i] << "\"";
            if (i != entry.second[1].size() - 1) jsonFile << ", ";
        }
        jsonFile << "]\n    }";
        if (++count != leadLastMap.size()) jsonFile << ",";
        jsonFile << "\n";
    }
    jsonFile << "  },\n";

    // Precedence Table
    map<string, map<string, string>> table = buildPrecedenceTableFromGrammar(filename, leadLastMap, startSymbol);
    set<string> terminals;
    for (const auto &row : table)
        terminals.insert(row.first);
    for (const auto &row : table)
        for (const auto &col : row.second)
            terminals.insert(col.first);

    jsonFile << "  \"precedence_table\": {\n";
    size_t rowCount = 0;
    for (const string &row : terminals) {
        jsonFile << "    \"" << row << "\": {";
        size_t colCount = 0;
        for (const string &col : terminals) {
            if (table[row].count(col)) {
                jsonFile << "\"" << col << "\": \"" << table[row][col] << "\"";
                if (++colCount != table[row].size()) jsonFile << ", ";
            }
        }
        jsonFile << "}";
        if (++rowCount != terminals.size()) jsonFile << ",";
        jsonFile << "\n";
    }
    jsonFile << "  }\n";
    jsonFile << "}\n";
    jsonFile.close();


    string input="";
    ifstream file("input.txt");
    getline(file, input);
    auto trace=  buildParseTable(table,input);
    saveTraceToJSON( trace, "outputparsing.json");
    return 0;
}
