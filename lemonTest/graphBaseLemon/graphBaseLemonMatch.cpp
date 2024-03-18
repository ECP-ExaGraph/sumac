#include <iostream>
#include <lemon/smart_graph.h>
#include <lemon/matching.h>
#include <lemon/lgf_reader.h>
#include <lemon/edge_set.h>
#include <lemon/concepts/graph_components.h>
#include <fstream>
#include "graph.hpp"
#include "types.hpp"
using namespace lemon;
using namespace std;
int main(int argc, char** argv)
{
    if(argc!=3){
        cout << argv[0] << " [outMatching] [LEMONbin]" << endl;
        return 1;
    }


    Graph* graph = nullptr;
    std::string inputFileName = argv[2];
    graph = new Graph(inputFileName);


    SmartGraph g;
    int numNodes = 67716231;
    int numEdges = 277557124;
    int currEdge = 0;

    SmartGraph::Edge* edgeList; 
    SmartGraph::Node* nodeList;
    float* weights;
    int* extMatch;
    long* srcs;
    long* dsts;


    nodeList = new SmartGraph::Node[numNodes];
    edgeList = new SmartGraph::Edge[numEdges];

    weights = new float[numEdges];
    extMatch = new int[numNodes];
    srcs = new long[numEdges];
    dsts = new long[numEdges];


    for(int i=0;i<numNodes;i++){
        nodeList[i] = g.addNode();
    }


    



    string compareInp(argv[1]); 

    ifstream matchinfile;
    matchinfile.open(compareInp);
    string mline; 
    int currMatch = 0;
    while(getline(matchinfile,mline)){
        extMatch[currMatch] = stoi(mline);
        currMatch++; 
    }

    matchinfile.close();

    printf("Finished here\n");
    GraphElem* indicesG;
    GraphElem* edgesG;
    GraphWeight* edgeWeightsG;

    indicesG = graph->get_index_ranges();
    edgesG = graph->get_edges();
    edgeWeightsG = graph->get_edge_weights();

    for(GraphElem i = 0;i<graph->get_num_vertices();i++){
        GraphElem adj1 = indicesG[i];
        GraphElem adj2 = indicesG[i+1];
        for(GraphElem j=adj1;j<adj2;j++){
            GraphElem currEdgeG = edgesG[j];
            GraphWeight currWeightG = edgeWeightsG[j];
            srcs[currEdge] = i;
            dsts[currEdge] = currEdgeG;
            weights[currEdge] = currWeightG;
            edgeList[currEdge] = g.addEdge(nodeList[i],nodeList[currEdgeG]);
            currEdge++;
        }
    }

/*
    while(getline(infile, line)) {
        //cout << line << endl;
        if(line[0]=='%'){
            flag = 1;
            continue;
        }
        if(flag == 1){
            flag = 0;
            continue;
        }
        string dl = " ";
        string token;
        int pos = 0;
        int v1;
        int v2;
        float weight; 
        int vcount = 0;
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
*/
    SmartGraph::EdgeMap<float> edgeWeights(g);
    for(int i=0;i<numEdges;i++){
        edgeWeights[edgeList[i]] = weights[i]; 
    }


    MaxWeightedMatching<SmartGraph,SmartGraph::EdgeMap<float>> match(g,edgeWeights);

    match.run();


    

    cout << "Graph: " << inputFileName << endl;
    cout << "Nodes: " << numNodes << " Edges: " << numEdges << endl;
    

    float matchWeightLemon = 0.0;
    float matchWeightExt = 0.0;

    for(int i=0;i<numNodes;i++){
        int flag = 0;
        int v1 = i;
        int v2L = g.id(match.mate(nodeList[i]));
        int v2E = extMatch[i];
        for(int j=0;j<numEdges;j++){
            if(flag==2){
                break;
            }
            if(srcs[j] == v1){
                if(dsts[j] == v2L){
                    matchWeightLemon += weights[j];
                    flag +=1;
                }
                if(dsts[j] == v2E){
                    matchWeightExt += weights[j];
                    flag+=1;
                }
            }
        } 
    }
    matchWeightLemon/=2;
    matchWeightExt/=2;
    cout << "Matching Weight Lemon: " << matchWeightLemon << endl;
    cout << "Matching Weight PC: " << matchWeightExt << endl;
    cout << "Matching Fraction: " << (matchWeightLemon - matchWeightExt)/matchWeightLemon << endl;




    //infile.close();
    free(nodeList);
    free(edgeList);
    free(weights);
    free(extMatch);
    free(srcs);
    free(dsts);
    return 0;

}