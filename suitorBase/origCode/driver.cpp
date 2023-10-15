#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <cuda.h>
#include <omp.h>




// Data structure to represent a CSR matrix
struct CSRMatrix {
    std::vector<double> values;
    std::vector<int> columns;
    std::vector<int> row_ptr;
    int numRows;
    int numCols;
    int n;
    int m;
};
void approx_match_gpu_suitor(int n, CSRMatrix csr);
// Function to read a Matrix Market file and convert it to CSR format
CSRMatrix matrixMarketToCSR(const std::string& filename, CSRMatrix& csrMatrix) {
    
    std::random_device rd;
    std::mt19937 gen(rd());

    // Define a distribution for the desired range
    double min_value = 0.0;
    double max_value = 1.0;
    std::uniform_real_distribution<double> distribution(min_value, max_value);
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return csrMatrix;
    }

    std::string line;
    int numNonZeros;
    bool headerRead = false;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%') {
            // Skip comments and empty lines
            continue;
        }

        if (!headerRead) {
            // Read header information
            std::istringstream headerStream(line);
            std::cout << line << std::endl;
            if (!(headerStream >> csrMatrix.numRows >> csrMatrix.numCols >> numNonZeros)) {
                std::cerr << "Invalid header format." << std::endl;
                return csrMatrix;
            }

            csrMatrix.row_ptr.resize(csrMatrix.numRows + 1, 0);
            headerRead = true;
        } else {
            // Read matrix data
            std::istringstream dataStream(line);
            int row, col;
            double value;

            if (!(dataStream >> row >> col)) {
                std::cerr << "Invalid data format." << std::endl;
                return csrMatrix;
            }

            // Adjust for 1-based indexing in Matrix Market format
            row--;
            col--;
            value = distribution(gen);
            // Store the data in CSR format
            csrMatrix.values.push_back(value);
            csrMatrix.columns.push_back(col);
            csrMatrix.row_ptr[row + 1]++;
        }
    }

    // Compute row_ptr
    for (int i = 1; i <= csrMatrix.numRows; ++i) {
        csrMatrix.row_ptr[i] += csrMatrix.row_ptr[i - 1];
    }
    csrMatrix.n = csrMatrix.row_ptr.size()-1;
    csrMatrix.m = csrMatrix.columns.size();
    std::cout << "Verts: " << csrMatrix.n << " Edges: "<< csrMatrix.m << std::endl;
    file.close();
    return csrMatrix;
}

int main(int argc, char** argv) {
    CSRMatrix csrMatrix;
    if(argc!=2){
        std::cout << argv[0] << " [mtxPath]" << std::endl;
        return 1;
    }
    std::string s(argv[1]);
    csrMatrix = matrixMarketToCSR(s, csrMatrix);
    approx_match_gpu_suitor(csrMatrix.n,csrMatrix);

    return 0;
}
