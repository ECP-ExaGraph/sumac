#include <iostream>
#include <fstream>
#include <vector>
#include <random>

struct CSRMatrix {
    std::vector<float> values;    // Non-zero values
    std::vector<int> indices;   // Column indices
    std::vector<int> edges;   // Row pointers

    int numRows;
    int numCols;
};

using namespace std;

int main(int argc, char** argv) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Define a distribution for the desired range
    double min_value = 0.0;
    double max_value = 1.0;
    std::uniform_real_distribution<double> distribution(min_value, max_value);

    // Open the Matrix Market file for reading
    if(argc!=2){
        cerr << argv[0] << " [pathToMM]" << endl;
        return 1;
    }
    string s(argv[1]);
    ifstream mmFile(s);

    // Check if the file is opened successfully
    if (!mmFile.is_open()) {
        cerr << "Failed to open the Matrix Market file." << endl;
        return 1;
    }
    string line;
    while (getline(mmFile, line)) {
        // Check if the line does not start with '%'
        if (line.empty() || line[0] != '%') {
            break; // Exit the loop if the condition is met
        }
    }

    char delimiter = ' ';

    size_t firstDelimPos = line.find(delimiter);
    string part1;
    string part2;
    string part3;

    cout << firstDelimPos << string::npos << endl;
    part1 = line.substr(0, firstDelimPos);
    size_t secondDelimPos = line.find(delimiter, firstDelimPos + 1);
    part2 = line.substr(firstDelimPos + 1, secondDelimPos - firstDelimPos - 1);
    part3 = line.substr(secondDelimPos + 1);

    cout << part1 << endl;
    int numRows, numCols, numNonZeros;
    numRows = stoi(part1);
    numCols = stoi(part2);
    numNonZeros = stoi(part3);
    cout << "ROWS: " << numRows << endl;
    // Create CSR matrix structure
    CSRMatrix csrMatrix;
    csrMatrix.numRows = numRows;
    csrMatrix.numCols = numCols;

    // Initialize row_ptr with zeros
    csrMatrix.edges.resize(numRows + 1, 0);

    // Read the matrix data and populate the CSR matrix
    for (int i = 0; i < numNonZeros; ++i) {
        int row, col;
        double value;
        mmFile >> row >> col;
        value = distribution(gen);

        // Adjust for 0-based indexing
        row--;
        col--;

        // Store the value and column index
        csrMatrix.values.push_back(value);
        csrMatrix.indices.push_back(col);

        // Update row_ptr
        csrMatrix.edges[row + 1]++;
    }

    // Cumulative sum in row_ptr
    for (int i = 1; i <= numRows; ++i) {
        csrMatrix.edges[i] += csrMatrix.edges[i - 1];
    }

    // Print the CSR matrix
    /*
    std::cout << "CSR Matrix:" << std::endl;
    for (int i = 0; i < numRows; ++i) {
        for (int j = csrMatrix.edges[i]; j < csrMatrix.edges[i + 1]; ++j) {
            std::cout << "Row: " << i << ", Column: " << csrMatrix.indices[j] << ", Value: " << csrMatrix.values[j] << std::endl;
        }
    }*/

    // Close the Matrix Market file
    mmFile.close();

    return 0;
}
