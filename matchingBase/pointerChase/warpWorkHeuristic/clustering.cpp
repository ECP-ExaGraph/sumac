#include <omp.h>
#include <cstdio>
#include "clustering.hpp"
void Clustering::singleton_partition()
{
    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel for
    for(GraphElem i = 0; i < nv_; ++i)
        commIds_[i] = i;
}

void Clustering::update_clustering(GraphElem* commIdsHost)
{
    omp_set_num_threads(omp_get_max_threads()); 
    #pragma omp parallel for
    for(GraphElem i = 0; i < nv_; ++i)
    {
        GraphElem myId = commIdsHost[commIds_[i]];
        commIds_[i] = myId;
    }
}

void Clustering::dump_partition(const std::string& filename, GraphElem* new_orders)
{
    if(new_orders != nullptr)
    {
        GraphElem* tmp = new GraphElem[nv_];
        #pragma omp parallel for
        for(GraphElem i = 0; i < nv_; ++i)
        {
            GraphElem c = commIds_[i];
            GraphElem id = new_orders[i];
            tmp[id] = c;
        }
        delete [] commIds_;
        commIds_ = tmp;
    }
    FILE* pFile;
    pFile = fopen(filename.c_str(), "wt");
    if(pFile != nullptr)
    {
        for(GraphElem i = 0; i < nv_; ++i)
            fprintf(pFile, "%lld\n", (long long)commIds_[i]);
    }
    fclose(pFile);
}