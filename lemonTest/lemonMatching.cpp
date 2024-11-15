#include <iostream>
#include <lemon/smart_graph.h>
#include <lemon/matching.h>
#include <lemon/lgf_reader.h>
#include <lemon/edge_set.h>
#include <lemon/concepts/graph_components.h>
#include <fstream>
using namespace lemon;
using namespace std;
int main(int argc, char** argv)
{

    if(argc!=3){
        cout << argv[0] << " [outMatching] [LEMONmtx]" << endl;
        return 1;
    }

    SmartGraph g;
    //int numNodes = 5000;
    int numNodes = 45101;
    int numEdges = 28967291;
    //int numEdges = 19994794;
    int currEdge = 0;

    SmartGraph::Edge* edgeList; 
    SmartGraph::Node* nodeList;
    double* weights;
    int* extMatch;
    long* srcs;
    long* dsts;


    nodeList = new SmartGraph::Node[numNodes];
    edgeList = new SmartGraph::Edge[numEdges];

    weights = new double[numEdges];
    extMatch = new int[numNodes];
    srcs = new long[numEdges];
    dsts = new long[numEdges];

    #pragma omp parallel for
    for(long i=0;i<numNodes;i++){
        nodeList[i] = g.addNode();
    }


    


    
    string compareInp(argv[1]); 
    string s(argv[2]);

    /*
    ifstream matchinfile;
    matchinfile.open(compareInp);
    string mline; 
    int currMatch = 0;
    while(getline(matchinfile,mline)){
        extMatch[currMatch] = stoi(mline);
        currMatch++; 
    }

    matchinfile.close();
    */
    printf("Finished here\n");
    ifstream infile;
    infile.open(s);
    string line;
    int flag = 0;


    


    while(getline(infile, line)) {
        //cout << line << endl;
        if(line[0]=='%'){
            //flag = 1;
            continue;
        }
        /*
        if(flag == 1){
            flag = 0;
            continue;
        }*/
        string dl = " ";
        string token;
        int pos = 0;
        long v1;
        long v2;
        float weight; 
        long vcount = 0;
        while ((pos = line.find(dl)) != std::string::npos) {
            token = line.substr(0, pos);
            //std::cout << token << std::endl;
            line.erase(0, pos + dl.length());
            if(vcount == 0){
                v1 = stoi(token);
                vcount++;
            }
            else{
                v2 = stoi(token);
                vcount = 0;
            }
            
        }
        weight = stof(line);
        weights[currEdge] = weight;
        srcs[currEdge] = v1-1;
        dsts[currEdge] = v2-1;
        edgeList[currEdge] = g.addEdge(nodeList[v1-1],nodeList[v2-1]);
        currEdge++;
    }

    SmartGraph::EdgeMap<float> edgeWeights(g);
    for(long i=0;i<numEdges;i++){
        edgeWeights[edgeList[i]] = weights[i]; 
    }


    MaxWeightedMatching<SmartGraph,SmartGraph::EdgeMap<float>> match(g,edgeWeights);

    cout << "Starting Matching" << endl; 
    match.run();


    

    cout << "Graph: " << s << endl;
    cout << "Nodes: " << numNodes << " Edges: " << numEdges << endl;
    
    
    float matchWeightLemon = 0.0;
    float matchWeightExt = 0.0;

    /*
    for(long i=0;i<numNodes;i++){
        long v1 = i;
        long v2L = g.id(match.mate(nodeList[i]));
        for(long j=0;j<numEdges;j++){
            if(srcs[j] == v1){
                if(dsts[j] == v2L){
                    matchWeightLemon += weights[j];
                }
                /*
                if(dsts[j] == v2E){
                    matchWeightExt += weights[j];
                    flag+=1;
                }s
            }
        } 
    }
    //matchWeightLemon/=2;
    matchWeightExt/=2;
    */
    cout << "Matching Weight Lemon: " << match.matchingWeight() << endl; 
    cout << "Matching Weight Lemon: " << matchWeightLemon << endl;
    //cout << "Matching Weight PC: " << matchWeightExt << endl;
    //cout << "Matching Fraction: " << (matchWeightLemon - matchWeightExt)/matchWeightLemon << endl;




    infile.close();
    free(nodeList);
    free(edgeList);
    free(weights);
    free(extMatch);
    free(srcs);
    free(dsts);
    return 0;

}